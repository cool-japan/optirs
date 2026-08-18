//! Regression tests for the F57 *wiring* fix: `EventCompressionEngine` is no
//! longer dead code reachable only from its own unit tests — with
//! `EventDrivenConfig::event_compression` enabled the live event queue really
//! stores compressed bytes, and events are reconstructed from those bytes on
//! the way out.
//!
//! Split into its own file (rather than an inline `#[cfg(test)] mod`) to keep
//! `event_driven.rs` under the workspace's 2000-line file-size policy.

use super::*;

fn compressed_config(algorithm: EventCompressionAlgorithm) -> EventDrivenConfig<f64> {
    EventDrivenConfig {
        event_compression: true,
        compression_algorithm: algorithm,
        // Rate limiting is orthogonal here and would reject bursts.
        rate_limits: HashMap::new(),
        temporal_correlation: false,
        ..Default::default()
    }
}

fn plain_config() -> EventDrivenConfig<f64> {
    EventDrivenConfig {
        event_compression: false,
        compression_algorithm: EventCompressionAlgorithm::None,
        rate_limits: HashMap::new(),
        temporal_correlation: false,
        ..Default::default()
    }
}

fn optimizer(config: EventDrivenConfig<f64>) -> EventDrivenOptimizer<f64> {
    EventDrivenOptimizer::new(
        config,
        STDPConfig::default(),
        MembraneDynamicsConfig::default(),
        8,
    )
}

/// Every field value below is exactly representable at the codec's
/// `EVENT_FIXED_POINT_SCALE` (1e-3) quantisation, so a correct round trip is
/// bit-exact and the differential assertions can use `==`.
fn stimulus(
    source: usize,
    value: f64,
    energy: f64,
    priority: EventPriority,
) -> NeuromorphicEvent<f64> {
    NeuromorphicEvent {
        event_type: EventType::ExternalStimulus,
        timestamp: 1.5,
        source_neuron: source,
        target_neuron: None,
        value,
        energy_cost: energy,
        priority,
    }
}

fn similar_spikes(count: usize) -> Vec<NeuromorphicEvent<f64>> {
    (0..count)
        .map(|i| NeuromorphicEvent {
            event_type: EventType::Spike,
            // Consecutive, closely-spaced events: exactly the case delta
            // encoding is supposed to shrink.
            timestamp: 10.0 + i as f64 * 0.5,
            source_neuron: 2 + i % 3,
            target_neuron: Some(5),
            value: 1.0,
            energy_cost: 0.05,
            priority: EventPriority::Normal,
        })
        .collect()
}

/// The live queue must hold compressed *bytes*, not whole events: the
/// uncompressed `BinaryHeap` stays empty while the compressed store grows.
#[test]
fn enqueue_stores_compressed_bytes_instead_of_events() {
    let mut opt = optimizer(compressed_config(EventCompressionAlgorithm::DeltaEncoding));

    for event in similar_spikes(8) {
        opt.enqueue_event(event).expect("enqueue");
    }

    assert!(
        opt.event_queue.is_empty(),
        "F57 wiring regression: events were stored uncompressed in the \
         in-memory priority queue even though event_compression is enabled"
    );
    assert_eq!(
        opt.get_queue_size(),
        8,
        "queued events must still be counted"
    );
    assert!(
        opt.compressed_queue_bytes() > 0,
        "F57 wiring regression: nothing was actually stored in compressed form"
    );
}

/// End-to-end round trip through the *live* path: enqueue -> compressed
/// storage -> `process_events` must leave exactly the same system state as
/// the uncompressed path for the identical event sequence.
#[test]
fn compressed_queue_round_trips_events_through_live_processing() {
    let events = vec![
        stimulus(0, 0.25, 0.05, EventPriority::Normal),
        stimulus(1, -0.5, 0.125, EventPriority::High),
        stimulus(0, 1.0, 0.25, EventPriority::Low),
        NeuromorphicEvent {
            event_type: EventType::TimerEvent,
            timestamp: 7.25,
            source_neuron: 3,
            target_neuron: Some(4),
            value: 0.0,
            energy_cost: 0.0,
            priority: EventPriority::Critical,
        },
    ];

    for algorithm in [
        EventCompressionAlgorithm::None,
        EventCompressionAlgorithm::DeltaEncoding,
        EventCompressionAlgorithm::SparseEncoding,
    ] {
        let mut plain = optimizer(plain_config());
        let mut compressed = optimizer(compressed_config(algorithm));

        for event in &events {
            plain.enqueue_event(event.clone()).expect("plain enqueue");
            compressed
                .enqueue_event(event.clone())
                .expect("compressed enqueue");
        }

        let plain_processed = plain.process_events().expect("plain process");
        let compressed_processed = compressed.process_events().expect("compressed process");

        assert_eq!(
            plain_processed, compressed_processed,
            "{algorithm:?}: compressed path processed a different number of events"
        );
        assert_eq!(
            plain.get_system_state().membrane_potentials,
            compressed.get_system_state().membrane_potentials,
            "{algorithm:?}: F57 wiring regression — events decoded from the \
             compressed queue did not reproduce the uncompressed membrane state"
        );
        assert_eq!(
            plain.get_system_state().current_time,
            compressed.get_system_state().current_time,
            "{algorithm:?}: TimerEvent timestamp did not survive the round trip"
        );
        assert_eq!(
            plain.get_metrics().energy_consumption,
            compressed.get_metrics().energy_consumption,
            "{algorithm:?}: per-event energy cost did not survive the round trip"
        );
        assert_eq!(
            compressed.get_queue_size(),
            0,
            "{algorithm:?}: compressed frames were left behind after processing"
        );
        assert_eq!(
            compressed.compressed_queue_bytes(),
            0,
            "{algorithm:?}: compressed byte accounting leaked after draining"
        );
    }
}

/// The compressed live queue must actually *compress*: for a stream of
/// similar consecutive events the bytes retained are fewer than the
/// uncompressed encoding of the same events.
#[test]
fn delta_compression_shrinks_the_live_queue() {
    let mut opt = optimizer(compressed_config(EventCompressionAlgorithm::DeltaEncoding));
    for event in similar_spikes(32) {
        opt.enqueue_event(event).expect("enqueue");
    }

    let (raw, compressed) = opt.compression_statistics();
    assert!(raw > 0, "no raw byte measurement was recorded");
    assert!(
        compressed < raw,
        "compressed live queue did not shrink the stream: raw={raw}, compressed={compressed}"
    );
    let ratio = opt.compression_ratio().expect("ratio after compressing");
    assert!(
        ratio < 1.0,
        "compression ratio must be below 1.0 for a similar-event stream, got {ratio}"
    );
    assert!(
        opt.compressed_queue_bytes() <= compressed,
        "stored bytes cannot exceed the cumulative compressed total"
    );
}

/// Priority scheduling must survive compression: one FIFO chain per priority
/// means the highest-priority frame still comes out first, and equal
/// priorities keep arrival order.
#[test]
fn compressed_queue_preserves_priority_then_fifo_order() {
    let mut opt = optimizer(compressed_config(EventCompressionAlgorithm::DeltaEncoding));

    // Enqueue deliberately out of priority order; `source_neuron` records
    // arrival order so FIFO can be checked within a priority level.
    let plan = [
        (EventPriority::Low, 0usize),
        (EventPriority::High, 1),
        (EventPriority::Normal, 2),
        (EventPriority::High, 3),
        (EventPriority::Critical, 4),
        (EventPriority::Low, 5),
    ];
    for (priority, source) in plan {
        opt.enqueue_event(stimulus(source, 0.5, 0.0, priority))
            .expect("enqueue");
    }

    let mut order = Vec::new();
    while let Some(event) = opt.pop_next_event().expect("pop") {
        order.push((event.priority, event.source_neuron));
    }

    assert_eq!(
        order,
        vec![
            (EventPriority::Critical, 4),
            (EventPriority::High, 1),
            (EventPriority::High, 3),
            (EventPriority::Normal, 2),
            (EventPriority::Low, 0),
            (EventPriority::Low, 5),
        ],
        "compressed queue lost priority-then-FIFO scheduling order"
    );
}

/// `max_queue_size` must count compressed frames, otherwise the capacity
/// guard silently stops working as soon as compression is enabled.
#[test]
fn queue_capacity_limit_counts_compressed_frames() {
    let mut config = compressed_config(EventCompressionAlgorithm::DeltaEncoding);
    config.max_queue_size = 4;
    let mut opt = optimizer(config);

    for (i, event) in similar_spikes(4).into_iter().enumerate() {
        opt.enqueue_event(event)
            .unwrap_or_else(|e| panic!("enqueue {i} within capacity failed: {e}"));
    }
    assert_eq!(opt.get_queue_size(), 4);
    assert!(
        opt.enqueue_event(stimulus(1, 0.5, 0.0, EventPriority::Normal))
            .is_err(),
        "queue capacity check ignored the compressed frames already queued"
    );
}

/// Clearing the queue must reset the delta codec baselines on both sides:
/// dropping frames breaks the chain, so an encoder left holding the baseline
/// of a discarded event would make every subsequent event decode wrong.
#[test]
fn clearing_the_compressed_queue_resets_codec_baselines() {
    let mut opt = optimizer(compressed_config(EventCompressionAlgorithm::DeltaEncoding));

    // Enqueue and then discard a burst with very different field values.
    for event in similar_spikes(3) {
        opt.enqueue_event(event).expect("enqueue burst");
    }
    opt.clear_event_queue();
    assert_eq!(opt.get_queue_size(), 0);
    assert_eq!(opt.compressed_queue_bytes(), 0);

    // The events that follow must decode exactly, i.e. identically to the
    // uncompressed path.
    let follow_up = vec![
        stimulus(0, 0.75, 0.05, EventPriority::Normal),
        stimulus(1, -0.25, 0.125, EventPriority::Normal),
    ];
    let mut plain = optimizer(plain_config());
    for event in &follow_up {
        opt.enqueue_event(event.clone()).expect("enqueue follow-up");
        plain.enqueue_event(event.clone()).expect("plain enqueue");
    }
    opt.process_events().expect("process compressed");
    plain.process_events().expect("process plain");

    assert_eq!(
        opt.get_system_state().membrane_potentials,
        plain.get_system_state().membrane_potentials,
        "codec baselines were not reset by clear_event_queue: events after a \
         clear decoded to different values than the uncompressed path"
    );
}

/// The measured event rate must be a real, finite events-per-second figure.
/// The previous code divided the processed count by an integer *millisecond*
/// elapsed time, which is zero for any batch finishing inside a clock tick,
/// pushing `inf`/`NaN` into the adaptive handler's history.
#[test]
fn measured_event_rate_is_finite_for_a_fast_batch() {
    let mut opt = optimizer(plain_config());
    for event in similar_spikes(4) {
        opt.enqueue_event(event).expect("enqueue");
    }
    let processed = opt.process_events().expect("process");
    assert_eq!(processed, 4);

    let rate = opt
        .last_measured_event_rate()
        .expect("a measurable interval must have been recorded");
    assert!(
        rate.is_finite(),
        "event processing rate must be finite, got {rate}"
    );
    assert!(
        rate > 0.0,
        "positive event count must yield a positive rate"
    );
    assert!(
        opt.current_adaptation_factor().is_finite(),
        "adaptation factor must stay finite"
    );
    assert!(
        opt.adaptive_batch_size() >= 1,
        "adaptive batch size must never be zero"
    );
}
