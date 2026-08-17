// Cross-Device Privacy Manager Module
//
// This module implements privacy management across devices and users in
// federated learning, covering the device -> user ownership mapping that
// user-level differential privacy depends on, per-device and per-user budget
// accounting, sliding-window participation frequency, and a temporal event
// log.
//
// Design notes (why the API looks the way it does)
// ------------------------------------------------
// An earlier revision of this file *fabricated* the two pieces of information
// that matter most:
//
//   * The device -> user mapping was invented on first sight of a client id
//     (`user_id: clientid.clone()`), which silently asserts "one device per
//     user". Under that assumption user-level DP degenerates to device-level
//     DP while still being reported as user-level, i.e. the privacy claim was
//     stronger than the accounting. Ownership must therefore be *declared*
//     via [`CrossDevicePrivacyManager::register_device`]; an unregistered
//     device is an error, never a guess.
//   * `participation_frequency` was an unbounded `+= 0.1` counter, so it grew
//     past 1.0 without bound and had no unit. A frequency is a rate: this
//     implementation measures
//     `|{rounds the device participated in} ∩ window| / |window|`, which is
//     always in `[0, 1]` and is exactly the sampling rate `q` that
//     amplification-by-subsampling accounting needs.
//
// User-level vs device-level accounting
// -------------------------------------
// When [`CrossDeviceConfig::user_level_privacy`] is set, the epsilon spent by
// a participating device is debited from a budget shared by every device that
// user owns (the standard user-level guarantee). When it is unset, each
// device carries its own budget. Both paths are enforced: a participation
// that would overspend is rejected rather than recorded.

use super::super::PrivacyBudget;
use crate::error::{OptimError, Result};
use chrono::Utc;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Default width, in federated rounds, of the sliding window over which
/// participation frequency is measured.
pub const DEFAULT_PARTICIPATION_WINDOW_ROUNDS: u64 = 100;

/// Default per-subject (device or user) epsilon budget.
pub const DEFAULT_SUBJECT_EPSILON_BUDGET: f64 = 1.0;

/// Cross-device privacy configuration
#[derive(Debug, Clone)]
pub struct CrossDeviceConfig {
    /// Charge epsilon against a budget shared by all of a user's devices
    /// (user-level DP) instead of against each device separately.
    pub user_level_privacy: bool,

    /// Maintain the location-cluster index used by geographic grouping
    /// queries.
    pub device_clustering: bool,

    /// Record a [`TemporalEvent`] for every participation so that temporal
    /// correlation across rounds can be audited.
    pub temporal_privacy: bool,

    /// Require every registration to declare a non-empty location cluster.
    pub geographic_privacy: bool,

    /// Require every registration to declare a non-empty demographic cohort.
    pub demographic_privacy: bool,

    /// Width, in rounds, of the participation-frequency window. Must be
    /// non-zero.
    pub participation_window_rounds: u64,

    /// Epsilon budget granted to each accounting subject: to each device when
    /// `user_level_privacy` is false, to each user when it is true.
    pub subject_epsilon_budget: f64,
}

impl Default for CrossDeviceConfig {
    fn default() -> Self {
        Self {
            user_level_privacy: false,
            device_clustering: false,
            temporal_privacy: false,
            geographic_privacy: false,
            demographic_privacy: false,
            participation_window_rounds: DEFAULT_PARTICIPATION_WINDOW_ROUNDS,
            subject_epsilon_budget: DEFAULT_SUBJECT_EPSILON_BUDGET,
        }
    }
}

/// Declaration of a device and the user that owns it.
///
/// This is the *only* way a device enters the manager. There is deliberately
/// no inference path: without a declared owner the manager cannot honestly
/// account user-level privacy, so it refuses to account at all.
#[derive(Debug, Clone, Default)]
pub struct DeviceRegistration {
    /// Stable identifier of the device (the federated client id).
    pub device_id: String,

    /// Identifier of the human/account that owns the device. Several devices
    /// may share one `user_id`; that is the whole point of user-level DP.
    pub user_id: String,

    /// Device class.
    pub device_type: DeviceType,

    /// Geographic/network cluster label. Required when
    /// [`CrossDeviceConfig::geographic_privacy`] is set.
    pub location_cluster: String,

    /// Demographic cohort label. Required when
    /// [`CrossDeviceConfig::demographic_privacy`] is set.
    pub demographic_cohort: String,
}

/// One participation of one device in one federated round.
#[derive(Debug, Clone)]
pub struct ParticipationRecord<T: Float + Debug + Send + Sync + 'static> {
    /// Federated round index. Must not go backwards for a given device.
    pub round: u64,

    /// Epsilon actually spent by this device in this round. Debited from the
    /// device's (or its user's) budget.
    pub epsilon_spent: f64,

    /// Observed L2 norm of the device's update, when the caller measured it.
    /// Used to maintain an empirical sensitivity estimate; `None` leaves the
    /// estimate untouched rather than inventing a value.
    pub update_l2_norm: Option<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> ParticipationRecord<T> {
    /// A participation that spends `epsilon_spent` in `round` and reports no
    /// update norm.
    pub fn new(round: u64, epsilon_spent: f64) -> Self {
        Self {
            round,
            epsilon_spent,
            update_l2_norm: None,
        }
    }

    /// Attach an observed update norm.
    pub fn with_update_norm(mut self, norm: T) -> Self {
        self.update_l2_norm = Some(norm);
        self
    }
}

/// Cross-device privacy manager
pub struct CrossDevicePrivacyManager<T: Float + Debug + Send + Sync + 'static> {
    config: CrossDeviceConfig,
    /// user_id -> owned device ids (sorted, deduplicated). Derived purely
    /// from declared registrations.
    user_clusters: HashMap<String, Vec<String>>,
    /// user_id -> shared budget, populated only under user-level privacy.
    user_budgets: HashMap<String, PrivacyBudget>,
    device_profiles: HashMap<String, DeviceProfile<T>>,
    temporal_correlations: HashMap<String, Vec<TemporalEvent>>,
}

/// Device profile for cross-device privacy
#[derive(Debug, Clone)]
pub struct DeviceProfile<T: Float + Debug + Send + Sync + 'static> {
    /// Declared device identifier.
    pub device_id: String,

    /// Declared owning user identifier.
    pub user_id: String,

    /// Declared device class.
    pub device_type: DeviceType,

    /// Declared location cluster (empty when geographic privacy is off and
    /// the caller did not supply one).
    pub location_cluster: String,

    /// Declared demographic cohort (empty when demographic privacy is off
    /// and the caller did not supply one).
    pub demographic_cohort: String,

    /// Sliding-window participation rate in `[0, 1]`: the number of rounds in
    /// the window in which this device participated, divided by the window
    /// width. Derived from [`Self::participation_rounds`]; do not set it
    /// directly.
    pub participation_frequency: f64,

    /// Rounds inside the current window in which the device participated,
    /// oldest first.
    pub participation_rounds: VecDeque<u64>,

    /// Total participations ever recorded, across all windows.
    pub total_participations: u64,

    /// Highest round for which a participation was recorded, if any.
    pub last_round: Option<u64>,

    /// Budget consumed by this specific device. Under user-level privacy the
    /// *enforced* budget is the user's; this field still tracks the device's
    /// own contribution to it.
    pub local_privacy_budget: PrivacyBudget,

    /// Empirical sensitivity estimate: the largest update norm observed so
    /// far. Zero until the caller reports one.
    pub sensitivity_estimate: T,
}

/// Device types for privacy analysis
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Default)]
pub enum DeviceType {
    /// Phones and tablets.
    #[default]
    Mobile,
    /// Laptops and workstations.
    Desktop,
    /// Constrained embedded devices.
    IoT,
    /// Edge/gateway nodes.
    Edge,
    /// Datacentre nodes.
    Server,
}

/// Temporal event for privacy tracking
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// Wall-clock time of the event, in seconds since the Unix epoch.
    pub timestamp: u64,

    /// Federated round the event belongs to.
    pub round: u64,

    /// What happened.
    pub event_type: TemporalEventType,

    /// Epsilon actually attributed to the event. Never a placeholder: for a
    /// participation this is the epsilon that was debited.
    pub privacy_impact: f64,
}

/// Kinds of [`TemporalEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalEventType {
    /// A device took part in a round.
    ClientParticipation,
    /// A model update was applied.
    ModelUpdate,
    /// Privacy budget was consumed outside of a participation.
    PrivacyBudgetConsumption,
    /// A secure aggregation completed.
    AggregationEvent,
}

impl<T: Float + Debug + Send + Sync + 'static> CrossDevicePrivacyManager<T> {
    /// Create a manager. Fails when the configuration is not usable, so that
    /// a nonsensical window width cannot silently produce a divide-by-zero
    /// frequency later.
    pub fn try_new(config: CrossDeviceConfig) -> Result<Self> {
        if config.participation_window_rounds == 0 {
            return Err(OptimError::InvalidConfig(
                "participation_window_rounds must be greater than zero".to_string(),
            ));
        }
        if !config.subject_epsilon_budget.is_finite() || config.subject_epsilon_budget <= 0.0 {
            return Err(OptimError::InvalidConfig(format!(
                "subject_epsilon_budget must be positive and finite, got {}",
                config.subject_epsilon_budget
            )));
        }
        Ok(Self {
            config,
            user_clusters: HashMap::new(),
            user_budgets: HashMap::new(),
            device_profiles: HashMap::new(),
            temporal_correlations: HashMap::new(),
        })
    }

    /// Create a manager, falling back to the default window width and budget
    /// when the supplied configuration specifies unusable values.
    ///
    /// Prefer [`Self::try_new`] in new code; this constructor exists so that
    /// the infallible signature used by existing callers keeps working.
    pub fn new(config: CrossDeviceConfig) -> Self {
        let mut sanitized = config;
        if sanitized.participation_window_rounds == 0 {
            sanitized.participation_window_rounds = DEFAULT_PARTICIPATION_WINDOW_ROUNDS;
        }
        if !sanitized.subject_epsilon_budget.is_finite() || sanitized.subject_epsilon_budget <= 0.0
        {
            sanitized.subject_epsilon_budget = DEFAULT_SUBJECT_EPSILON_BUDGET;
        }
        Self {
            config: sanitized,
            user_clusters: HashMap::new(),
            user_budgets: HashMap::new(),
            device_profiles: HashMap::new(),
            temporal_correlations: HashMap::new(),
        }
    }

    /// Declare a device and its owning user.
    ///
    /// Re-registering the same `(device_id, user_id)` pair refreshes the
    /// device's descriptive labels and is idempotent with respect to budget
    /// and participation history. Re-registering a device under a *different*
    /// user is rejected: silently re-parenting a device would retroactively
    /// invalidate every user-level epsilon already accounted for it.
    pub fn register_device(&mut self, registration: DeviceRegistration) -> Result<()> {
        if registration.device_id.is_empty() {
            return Err(OptimError::InvalidParameter(
                "device_id must not be empty".to_string(),
            ));
        }
        if registration.user_id.is_empty() {
            return Err(OptimError::InvalidParameter(format!(
                "device {} must declare a non-empty owning user_id; user-level privacy \
                 cannot be accounted for an unknown owner",
                registration.device_id
            )));
        }
        if self.config.geographic_privacy && registration.location_cluster.is_empty() {
            return Err(OptimError::InvalidParameter(format!(
                "geographic_privacy is enabled, so device {} must declare a location_cluster",
                registration.device_id
            )));
        }
        if self.config.demographic_privacy && registration.demographic_cohort.is_empty() {
            return Err(OptimError::InvalidParameter(format!(
                "demographic_privacy is enabled, so device {} must declare a demographic_cohort",
                registration.device_id
            )));
        }

        if let Some(existing) = self.device_profiles.get_mut(&registration.device_id) {
            if existing.user_id != registration.user_id {
                return Err(OptimError::InvalidParameter(format!(
                    "device {} is already owned by user {}; refusing to re-parent it to {}",
                    registration.device_id, existing.user_id, registration.user_id
                )));
            }
            existing.device_type = registration.device_type;
            existing.location_cluster = registration.location_cluster;
            existing.demographic_cohort = registration.demographic_cohort;
            return Ok(());
        }

        let profile = DeviceProfile {
            device_id: registration.device_id.clone(),
            user_id: registration.user_id.clone(),
            device_type: registration.device_type,
            location_cluster: registration.location_cluster,
            demographic_cohort: registration.demographic_cohort,
            participation_frequency: 0.0,
            participation_rounds: VecDeque::new(),
            total_participations: 0,
            last_round: None,
            local_privacy_budget: self.fresh_budget(),
            sensitivity_estimate: T::zero(),
        };
        self.device_profiles
            .insert(registration.device_id.clone(), profile);

        let owned = self
            .user_clusters
            .entry(registration.user_id.clone())
            .or_default();
        if !owned.contains(&registration.device_id) {
            owned.push(registration.device_id);
            owned.sort();
        }
        if self.config.user_level_privacy {
            let budget = self.fresh_budget();
            self.user_budgets
                .entry(registration.user_id)
                .or_insert(budget);
        }
        Ok(())
    }

    /// Record that `device_id` participated in a round.
    ///
    /// Enforces, in order: the device is registered, the record is
    /// well-formed, the round does not go backwards, and the epsilon fits in
    /// the enforced budget. Only then is any state mutated, so a rejected
    /// participation leaves the manager exactly as it was.
    pub fn record_participation(
        &mut self,
        device_id: &str,
        record: ParticipationRecord<T>,
    ) -> Result<()> {
        if !record.epsilon_spent.is_finite() || record.epsilon_spent < 0.0 {
            return Err(OptimError::InvalidParameter(format!(
                "epsilon_spent must be finite and non-negative, got {}",
                record.epsilon_spent
            )));
        }

        let (owner, last_round) = {
            let profile = self.device_profiles.get(device_id).ok_or_else(|| {
                OptimError::InvalidParameter(format!(
                    "device {device_id} is not registered; call register_device with its \
                     owning user_id before recording participation"
                ))
            })?;
            (profile.user_id.clone(), profile.last_round)
        };

        if let Some(previous) = last_round {
            if record.round < previous {
                return Err(OptimError::InvalidParameter(format!(
                    "round {} for device {device_id} is earlier than the last recorded round {}",
                    record.round, previous
                )));
            }
        }

        // Budget check before any mutation.
        if self.config.user_level_privacy {
            let remaining = self
                .user_budgets
                .get(&owner)
                .map(|budget| budget.epsilon_remaining)
                .unwrap_or(self.config.subject_epsilon_budget);
            if record.epsilon_spent > remaining {
                return Err(OptimError::PrivacyAccountingError(format!(
                    "user {owner} has {remaining} epsilon remaining; device {device_id} \
                     requested {}",
                    record.epsilon_spent
                )));
            }
        } else {
            let remaining = self
                .device_profiles
                .get(device_id)
                .map(|profile| profile.local_privacy_budget.epsilon_remaining)
                .unwrap_or(0.0);
            if record.epsilon_spent > remaining {
                return Err(OptimError::PrivacyAccountingError(format!(
                    "device {device_id} has {remaining} epsilon remaining; requested {}",
                    record.epsilon_spent
                )));
            }
        }

        let window = self.config.participation_window_rounds;
        let oldest_in_window = record.round.saturating_sub(window.saturating_sub(1));

        let profile = self.device_profiles.get_mut(device_id).ok_or_else(|| {
            OptimError::InvalidState(format!("device {device_id} vanished during accounting"))
        })?;

        // Sliding window: drop rounds that fell out of it, then admit this one
        // (a repeated round counts once, since a device participates in a
        // round at most once).
        while profile
            .participation_rounds
            .front()
            .is_some_and(|&r| r < oldest_in_window)
        {
            profile.participation_rounds.pop_front();
        }
        if profile.participation_rounds.back() != Some(&record.round) {
            profile.participation_rounds.push_back(record.round);
        }
        profile.participation_frequency = profile.participation_rounds.len() as f64 / window as f64;
        profile.total_participations = profile.total_participations.saturating_add(1);
        profile.last_round = Some(record.round);

        profile.local_privacy_budget.epsilon_consumed += record.epsilon_spent;
        profile.local_privacy_budget.epsilon_remaining =
            (profile.local_privacy_budget.epsilon_remaining - record.epsilon_spent).max(0.0);
        profile.local_privacy_budget.steps_taken =
            profile.local_privacy_budget.steps_taken.saturating_add(1);

        if let Some(norm) = record.update_l2_norm {
            if norm > profile.sensitivity_estimate {
                profile.sensitivity_estimate = norm;
            }
        }

        if self.config.user_level_privacy {
            let fresh = self.fresh_budget();
            let budget = self.user_budgets.entry(owner).or_insert(fresh);
            budget.epsilon_consumed += record.epsilon_spent;
            budget.epsilon_remaining = (budget.epsilon_remaining - record.epsilon_spent).max(0.0);
            budget.steps_taken = budget.steps_taken.saturating_add(1);
        }

        if self.config.temporal_privacy {
            self.temporal_correlations
                .entry(device_id.to_string())
                .or_default()
                .push(TemporalEvent {
                    timestamp: Self::unix_now(),
                    round: record.round,
                    event_type: TemporalEventType::ClientParticipation,
                    privacy_impact: record.epsilon_spent,
                });
        }

        Ok(())
    }

    /// Get device profile for a device.
    pub fn get_device_profile(&self, device_id: &str) -> Option<&DeviceProfile<T>> {
        self.device_profiles.get(device_id)
    }

    /// The declared owner of `device_id`, or `None` when it is unregistered.
    pub fn user_of_device(&self, device_id: &str) -> Option<&str> {
        self.device_profiles
            .get(device_id)
            .map(|profile| profile.user_id.as_str())
    }

    /// Get the temporal event log for a device.
    pub fn get_temporal_correlations(&self, device_id: &str) -> Option<&Vec<TemporalEvent>> {
        self.temporal_correlations.get(device_id)
    }

    /// Declare that `user_id` owns exactly `device_ids`.
    ///
    /// Every listed device must already be registered to that user. This is a
    /// *consistency assertion*, not a way to invent ownership: the mapping it
    /// installs is the one `register_device` already built.
    pub fn create_user_cluster(&mut self, user_id: String, device_ids: Vec<String>) -> Result<()> {
        for device_id in device_ids.iter() {
            match self.device_profiles.get(device_id) {
                None => {
                    return Err(OptimError::InvalidParameter(format!(
                        "device {device_id} is not registered, so it cannot be placed in the \
                         cluster of user {user_id}"
                    )));
                }
                Some(profile) if profile.user_id != user_id => {
                    return Err(OptimError::InvalidParameter(format!(
                        "device {device_id} is owned by user {}, not {user_id}",
                        profile.user_id
                    )));
                }
                Some(_) => {}
            }
        }
        let mut sorted = device_ids;
        sorted.sort();
        sorted.dedup();
        self.user_clusters.insert(user_id, sorted);
        Ok(())
    }

    /// Devices owned by `user_id`, as built from declared registrations.
    pub fn get_user_cluster(&self, user_id: &str) -> Option<&Vec<String>> {
        self.user_clusters.get(user_id)
    }

    /// The enforced budget for `user_id` under user-level privacy.
    ///
    /// Returns an error when user-level privacy is disabled, because in that
    /// mode no user-level budget exists and returning a device budget in its
    /// place would misreport the guarantee.
    pub fn user_privacy_budget(&self, user_id: &str) -> Result<&PrivacyBudget> {
        if !self.config.user_level_privacy {
            return Err(OptimError::InvalidState(
                "user_level_privacy is disabled; budgets are enforced per device, so there is \
                 no user-level budget to report"
                    .to_string(),
            ));
        }
        self.user_budgets.get(user_id).ok_or_else(|| {
            OptimError::InvalidParameter(format!("user {user_id} has no registered device"))
        })
    }

    /// Registered devices sharing `location_cluster`, sorted.
    ///
    /// Requires [`CrossDeviceConfig::device_clustering`]; without it the
    /// index is not maintained and an empty answer would be misleading.
    pub fn devices_in_location_cluster(&self, location_cluster: &str) -> Result<Vec<String>> {
        if !self.config.device_clustering {
            return Err(OptimError::InvalidState(
                "device_clustering is disabled; enable it to query location clusters".to_string(),
            ));
        }
        let mut members: Vec<String> = self
            .device_profiles
            .values()
            .filter(|profile| profile.location_cluster == location_cluster)
            .map(|profile| profile.device_id.clone())
            .collect();
        members.sort();
        Ok(members)
    }

    /// Check if user-level privacy is enabled
    pub fn is_user_level_privacy_enabled(&self) -> bool {
        self.config.user_level_privacy
    }

    /// Check if device clustering is enabled
    pub fn is_device_clustering_enabled(&self) -> bool {
        self.config.device_clustering
    }

    /// Check if temporal privacy is enabled
    pub fn is_temporal_privacy_enabled(&self) -> bool {
        self.config.temporal_privacy
    }

    /// Get configuration
    pub fn config(&self) -> &CrossDeviceConfig {
        &self.config
    }

    /// Get number of registered devices
    pub fn device_count(&self) -> usize {
        self.device_profiles.len()
    }

    /// Number of distinct users owning at least one registered device.
    pub fn user_count(&self) -> usize {
        self.user_clusters.len()
    }

    /// Sliding-window participation rate of a device, in `[0, 1]`.
    ///
    /// This is the sampling probability `q` for that device over the
    /// configured window, not a running count.
    pub fn get_participation_frequency(&self, device_id: &str) -> Option<f64> {
        self.device_profiles
            .get(device_id)
            .map(|profile| profile.participation_frequency)
    }

    /// Participation rate a device *would* report if the window were
    /// evaluated as of `round`, without mutating any state.
    ///
    /// Frequency decays as rounds pass without participation; this lets a
    /// caller observe that decay before the device is next selected.
    pub fn participation_frequency_at(&self, device_id: &str, round: u64) -> Option<f64> {
        let profile = self.device_profiles.get(device_id)?;
        let window = self.config.participation_window_rounds;
        let oldest = round.saturating_sub(window.saturating_sub(1));
        let in_window = profile
            .participation_rounds
            .iter()
            .filter(|&&r| r >= oldest && r <= round)
            .count();
        Some(in_window as f64 / window as f64)
    }

    fn fresh_budget(&self) -> PrivacyBudget {
        PrivacyBudget {
            epsilon_consumed: 0.0,
            epsilon_remaining: self.config.subject_epsilon_budget,
            steps_taken: 0,
            ..PrivacyBudget::default()
        }
    }

    fn unix_now() -> u64 {
        let seconds = Utc::now().timestamp();
        if seconds < 0 {
            0
        } else {
            seconds as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registration(device_id: &str, user_id: &str) -> DeviceRegistration {
        DeviceRegistration {
            device_id: device_id.to_string(),
            user_id: user_id.to_string(),
            device_type: DeviceType::Mobile,
            location_cluster: String::new(),
            demographic_cohort: String::new(),
        }
    }

    fn manager(config: CrossDeviceConfig) -> CrossDevicePrivacyManager<f64> {
        CrossDevicePrivacyManager::<f64>::try_new(config).expect("valid config")
    }

    #[test]
    fn test_cross_device_config() {
        let config = CrossDeviceConfig::default();
        assert!(!config.user_level_privacy);
        assert!(!config.device_clustering);
        assert!(!config.temporal_privacy);
        assert_eq!(
            config.participation_window_rounds,
            DEFAULT_PARTICIPATION_WINDOW_ROUNDS
        );
    }

    #[test]
    fn test_device_profile_creation() {
        let mut mgr = manager(CrossDeviceConfig::default());
        mgr.register_device(DeviceRegistration {
            device_id: "device_1".to_string(),
            user_id: "user_1".to_string(),
            device_type: DeviceType::Mobile,
            location_cluster: "cluster_a".to_string(),
            demographic_cohort: String::new(),
        })
        .expect("registration");

        let profile = mgr.get_device_profile("device_1").expect("profile");
        assert_eq!(profile.device_id, "device_1");
        assert_eq!(profile.user_id, "user_1");
        assert_eq!(profile.location_cluster, "cluster_a");
        assert!(matches!(profile.device_type, DeviceType::Mobile));
        // Sensitivity starts at zero, not at a fabricated 1.0.
        assert_eq!(profile.sensitivity_estimate, 0.0);
    }

    // ---------------------------------------------------------------------
    // F38/F39: the device -> user mapping is declared, never invented.
    // ---------------------------------------------------------------------

    #[test]
    fn unregistered_device_participation_is_rejected_not_auto_mapped() {
        let mut mgr = manager(CrossDeviceConfig::default());
        let err = mgr
            .record_participation("ghost", ParticipationRecord::new(1, 0.01))
            .expect_err("unregistered device must be refused");
        assert!(
            format!("{err}").contains("not registered"),
            "error should say the device is unregistered, got: {err}"
        );
        // Crucially: no profile was fabricated as a side effect.
        assert_eq!(mgr.device_count(), 0);
        assert!(mgr.get_device_profile("ghost").is_none());
        assert!(mgr.user_of_device("ghost").is_none());
    }

    #[test]
    fn several_devices_can_share_one_user() {
        let mut mgr = manager(CrossDeviceConfig::default());
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        mgr.register_device(registration("laptop", "alice"))
            .expect("laptop");
        mgr.register_device(registration("tablet", "bob"))
            .expect("tablet");

        assert_eq!(mgr.device_count(), 3);
        assert_eq!(mgr.user_count(), 2);
        assert_eq!(mgr.user_of_device("phone"), Some("alice"));
        assert_eq!(mgr.user_of_device("laptop"), Some("alice"));
        assert_eq!(mgr.user_of_device("tablet"), Some("bob"));
        assert_eq!(
            mgr.get_user_cluster("alice"),
            Some(&vec!["laptop".to_string(), "phone".to_string()])
        );
        assert_eq!(
            mgr.get_user_cluster("bob"),
            Some(&vec!["tablet".to_string()])
        );
    }

    #[test]
    fn device_cannot_be_silently_reparented() {
        let mut mgr = manager(CrossDeviceConfig::default());
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        let err = mgr
            .register_device(registration("phone", "bob"))
            .expect_err("re-parenting must fail");
        assert!(format!("{err}").contains("already owned by user alice"));
        assert_eq!(mgr.user_of_device("phone"), Some("alice"));
    }

    #[test]
    fn empty_user_id_is_rejected() {
        let mut mgr = manager(CrossDeviceConfig::default());
        let err = mgr
            .register_device(registration("phone", ""))
            .expect_err("empty owner must fail");
        assert!(format!("{err}").contains("non-empty owning user_id"));
    }

    #[test]
    fn create_user_cluster_rejects_devices_it_does_not_own() {
        let mut mgr = manager(CrossDeviceConfig::default());
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        mgr.register_device(registration("tablet", "bob"))
            .expect("tablet");

        assert!(mgr
            .create_user_cluster("alice".to_string(), vec!["phone".to_string()])
            .is_ok());
        let err = mgr
            .create_user_cluster("alice".to_string(), vec!["tablet".to_string()])
            .expect_err("foreign device must be refused");
        assert!(format!("{err}").contains("owned by user bob"));
        let err = mgr
            .create_user_cluster("alice".to_string(), vec!["nope".to_string()])
            .expect_err("unknown device must be refused");
        assert!(format!("{err}").contains("not registered"));
    }

    #[test]
    fn geographic_and_demographic_privacy_require_their_labels() {
        let mut mgr = manager(CrossDeviceConfig {
            geographic_privacy: true,
            ..CrossDeviceConfig::default()
        });
        let err = mgr
            .register_device(registration("phone", "alice"))
            .expect_err("missing location cluster must fail");
        assert!(format!("{err}").contains("location_cluster"));

        let mut mgr = manager(CrossDeviceConfig {
            demographic_privacy: true,
            ..CrossDeviceConfig::default()
        });
        let err = mgr
            .register_device(registration("phone", "alice"))
            .expect_err("missing cohort must fail");
        assert!(format!("{err}").contains("demographic_cohort"));
    }

    // ---------------------------------------------------------------------
    // F40: participation_frequency is a windowed rate in [0, 1].
    // ---------------------------------------------------------------------

    #[test]
    fn participation_frequency_is_a_bounded_windowed_rate() {
        let window = 10_u64;
        let mut mgr = manager(CrossDeviceConfig {
            participation_window_rounds: window,
            subject_epsilon_budget: 1000.0,
            ..CrossDeviceConfig::default()
        });
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");

        // Participate in every one of 200 rounds. The old implementation
        // returned 1.0 + 0.1 * 199 = 20.9; a rate cannot exceed 1.0.
        for round in 1..=200_u64 {
            mgr.record_participation("phone", ParticipationRecord::new(round, 0.001))
                .expect("participation");
        }
        let frequency = mgr.get_participation_frequency("phone").expect("frequency");
        assert!(
            (frequency - 1.0).abs() < 1e-12,
            "participating in every round must give rate 1.0, got {frequency}"
        );

        let profile = mgr.get_device_profile("phone").expect("profile");
        assert_eq!(profile.total_participations, 200);
        // The window is bounded: only `window` rounds are retained.
        assert_eq!(profile.participation_rounds.len(), window as usize);
    }

    #[test]
    fn participation_frequency_matches_the_fraction_of_rounds_in_the_window() {
        let window = 10_u64;
        let mut mgr = manager(CrossDeviceConfig {
            participation_window_rounds: window,
            subject_epsilon_budget: 100.0,
            ..CrossDeviceConfig::default()
        });
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");

        // Rounds 1..=20, participating on every other round: rounds
        // 11,13,15,17,19 fall inside the window [10, 19] as of round 19, so
        // the rate is 5/10.
        for round in (1..=19_u64).step_by(2) {
            mgr.record_participation("phone", ParticipationRecord::new(round, 0.01))
                .expect("participation");
        }
        let frequency = mgr.get_participation_frequency("phone").expect("frequency");
        assert!(
            (frequency - 0.5).abs() < 1e-12,
            "expected 5/10 = 0.5, got {frequency}"
        );
    }

    #[test]
    fn participation_frequency_decays_as_the_window_moves_past() {
        let window = 5_u64;
        let mut mgr = manager(CrossDeviceConfig {
            participation_window_rounds: window,
            ..CrossDeviceConfig::default()
        });
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        mgr.record_participation("phone", ParticipationRecord::new(1, 0.01))
            .expect("participation");

        assert_eq!(mgr.participation_frequency_at("phone", 1), Some(0.2));
        assert_eq!(mgr.participation_frequency_at("phone", 5), Some(0.2));
        // Round 1 has fallen out of the window [6, 10].
        assert_eq!(mgr.participation_frequency_at("phone", 10), Some(0.0));
    }

    #[test]
    fn repeated_round_counts_once() {
        let mut mgr = manager(CrossDeviceConfig {
            participation_window_rounds: 10,
            ..CrossDeviceConfig::default()
        });
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        mgr.record_participation("phone", ParticipationRecord::new(3, 0.01))
            .expect("first");
        mgr.record_participation("phone", ParticipationRecord::new(3, 0.01))
            .expect("second");

        let profile = mgr.get_device_profile("phone").expect("profile");
        assert_eq!(profile.participation_rounds.len(), 1);
        assert!((profile.participation_frequency - 0.1).abs() < 1e-12);
        // Both spends are still charged.
        assert!((profile.local_privacy_budget.epsilon_consumed - 0.02).abs() < 1e-12);
    }

    #[test]
    fn rounds_may_not_go_backwards() {
        let mut mgr = manager(CrossDeviceConfig::default());
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        mgr.record_participation("phone", ParticipationRecord::new(10, 0.01))
            .expect("round 10");
        let err = mgr
            .record_participation("phone", ParticipationRecord::new(9, 0.01))
            .expect_err("going back must fail");
        assert!(format!("{err}").contains("earlier than the last recorded round"));
    }

    // ---------------------------------------------------------------------
    // Budget accounting is enforced, in both modes.
    // ---------------------------------------------------------------------

    #[test]
    fn device_level_budget_is_enforced_per_device() {
        let mut mgr = manager(CrossDeviceConfig {
            subject_epsilon_budget: 0.05,
            ..CrossDeviceConfig::default()
        });
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        mgr.register_device(registration("laptop", "alice"))
            .expect("laptop");

        mgr.record_participation("phone", ParticipationRecord::new(1, 0.04))
            .expect("first spend fits");
        let err = mgr
            .record_participation("phone", ParticipationRecord::new(2, 0.04))
            .expect_err("overspend must be refused");
        assert!(format!("{err}").contains("epsilon remaining"));

        // The refusal did not mutate anything.
        let profile = mgr.get_device_profile("phone").expect("profile");
        assert!((profile.local_privacy_budget.epsilon_consumed - 0.04).abs() < 1e-12);
        assert_eq!(profile.last_round, Some(1));

        // The sibling device has its own untouched budget.
        mgr.record_participation("laptop", ParticipationRecord::new(2, 0.04))
            .expect("sibling has its own budget");
    }

    #[test]
    fn user_level_budget_is_shared_across_a_users_devices() {
        let mut mgr = manager(CrossDeviceConfig {
            user_level_privacy: true,
            subject_epsilon_budget: 0.05,
            ..CrossDeviceConfig::default()
        });
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        mgr.register_device(registration("laptop", "alice"))
            .expect("laptop");
        mgr.register_device(registration("tablet", "bob"))
            .expect("tablet");

        mgr.record_participation("phone", ParticipationRecord::new(1, 0.04))
            .expect("phone spend");
        // The laptop shares alice's budget, so it may not spend 0.04 more.
        let err = mgr
            .record_participation("laptop", ParticipationRecord::new(2, 0.04))
            .expect_err("shared budget must be enforced");
        assert!(format!("{err}").contains("user alice"));

        let alice = mgr.user_privacy_budget("alice").expect("alice budget");
        assert!((alice.epsilon_consumed - 0.04).abs() < 1e-12);
        assert!((alice.epsilon_remaining - 0.01).abs() < 1e-12);

        // Bob is unaffected.
        mgr.record_participation("tablet", ParticipationRecord::new(2, 0.04))
            .expect("bob spend");
        let bob = mgr.user_privacy_budget("bob").expect("bob budget");
        assert!((bob.epsilon_consumed - 0.04).abs() < 1e-12);
    }

    #[test]
    fn user_budget_query_errors_when_user_level_privacy_is_off() {
        let mut mgr = manager(CrossDeviceConfig::default());
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        let err = mgr
            .user_privacy_budget("alice")
            .expect_err("no user-level budget exists");
        assert!(format!("{err}").contains("user_level_privacy is disabled"));
    }

    // ---------------------------------------------------------------------
    // Sensitivity, temporal log and clustering.
    // ---------------------------------------------------------------------

    #[test]
    fn sensitivity_estimate_tracks_the_largest_observed_norm() {
        let mut mgr = manager(CrossDeviceConfig::default());
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");

        mgr.record_participation(
            "phone",
            ParticipationRecord::new(1, 0.01).with_update_norm(0.5),
        )
        .expect("round 1");
        assert_eq!(
            mgr.get_device_profile("phone")
                .map(|p| p.sensitivity_estimate),
            Some(0.5)
        );

        mgr.record_participation(
            "phone",
            ParticipationRecord::new(2, 0.01).with_update_norm(2.25),
        )
        .expect("round 2");
        assert_eq!(
            mgr.get_device_profile("phone")
                .map(|p| p.sensitivity_estimate),
            Some(2.25)
        );

        // A smaller norm does not shrink the estimate, and omitting the norm
        // leaves it alone rather than resetting it to a default.
        mgr.record_participation(
            "phone",
            ParticipationRecord::new(3, 0.01).with_update_norm(0.1),
        )
        .expect("round 3");
        mgr.record_participation("phone", ParticipationRecord::new(4, 0.01))
            .expect("round 4");
        assert_eq!(
            mgr.get_device_profile("phone")
                .map(|p| p.sensitivity_estimate),
            Some(2.25)
        );
    }

    #[test]
    fn temporal_events_record_the_real_epsilon_and_round() {
        let mut mgr = manager(CrossDeviceConfig {
            temporal_privacy: true,
            ..CrossDeviceConfig::default()
        });
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        mgr.record_participation("phone", ParticipationRecord::new(1, 0.02))
            .expect("round 1");
        mgr.record_participation("phone", ParticipationRecord::new(7, 0.03))
            .expect("round 7");

        let events = mgr.get_temporal_correlations("phone").expect("events");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].round, 1);
        assert_eq!(events[1].round, 7);
        // privacy_impact is the epsilon actually debited, not a hardcoded 1.0.
        assert!((events[0].privacy_impact - 0.02).abs() < 1e-12);
        assert!((events[1].privacy_impact - 0.03).abs() < 1e-12);
        // Timestamps are real wall-clock seconds, well past the epoch.
        assert!(events[0].timestamp > 1_600_000_000);
        assert!(events[1].timestamp >= events[0].timestamp);
    }

    #[test]
    fn temporal_log_is_not_kept_when_temporal_privacy_is_off() {
        let mut mgr = manager(CrossDeviceConfig::default());
        mgr.register_device(registration("phone", "alice"))
            .expect("phone");
        mgr.record_participation("phone", ParticipationRecord::new(1, 0.02))
            .expect("round 1");
        assert!(mgr.get_temporal_correlations("phone").is_none());
    }

    #[test]
    fn location_clusters_group_real_registrations() {
        let mut mgr = manager(CrossDeviceConfig {
            device_clustering: true,
            geographic_privacy: true,
            ..CrossDeviceConfig::default()
        });
        for (device, user, cluster) in [
            ("phone", "alice", "eu-west"),
            ("laptop", "alice", "eu-west"),
            ("tablet", "bob", "us-east"),
        ] {
            mgr.register_device(DeviceRegistration {
                device_id: device.to_string(),
                user_id: user.to_string(),
                device_type: DeviceType::Mobile,
                location_cluster: cluster.to_string(),
                demographic_cohort: String::new(),
            })
            .expect("registration");
        }

        assert_eq!(
            mgr.devices_in_location_cluster("eu-west")
                .expect("cluster query"),
            vec!["laptop".to_string(), "phone".to_string()]
        );
        assert_eq!(
            mgr.devices_in_location_cluster("us-east")
                .expect("cluster query"),
            vec!["tablet".to_string()]
        );
    }

    #[test]
    fn location_cluster_query_errors_when_clustering_is_off() {
        let mgr = manager(CrossDeviceConfig::default());
        let err = mgr
            .devices_in_location_cluster("eu-west")
            .expect_err("clustering disabled");
        assert!(format!("{err}").contains("device_clustering is disabled"));
    }

    #[test]
    fn zero_window_is_rejected_by_try_new_and_sanitized_by_new() {
        let bad = CrossDeviceConfig {
            participation_window_rounds: 0,
            ..CrossDeviceConfig::default()
        };
        assert!(CrossDevicePrivacyManager::<f64>::try_new(bad.clone()).is_err());
        let mgr = CrossDevicePrivacyManager::<f64>::new(bad);
        assert_eq!(
            mgr.config().participation_window_rounds,
            DEFAULT_PARTICIPATION_WINDOW_ROUNDS
        );
    }
}
