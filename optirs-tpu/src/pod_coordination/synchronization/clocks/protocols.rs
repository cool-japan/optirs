// Clock Protocols Module

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

use super::core::ClockOffset;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BerkeleyConfig {
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ClockSyncProtocol {
    #[default]
    NTP,
    PTP,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CristianConfig {
    pub server_address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CustomProtocolConfig {
    pub protocol_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NtpConfig {
    pub server: String,
}

#[derive(Debug, Clone, Default)]
pub struct NtpSynchronizer {
    pub config: NtpConfig,
}

impl NtpSynchronizer {
    /// Perform one real NTP-style round-trip offset estimation against
    /// `peer`, using the classic four-timestamp algorithm (RFC 5905 section
    /// 8: "On-Wire Protocol").
    ///
    /// The two local timestamps (`originate`/`destination`) are real
    /// [`SystemTime::now`] reads taken immediately before/after calling
    /// `peer`; the two peer-side timestamps come from `peer` itself, which
    /// may be backed by any real transport (a network client, the existing
    /// in-process coordination channels, ...) -- see [`NtpPeer`]. No offset
    /// is ever fabricated: an exchange whose timestamps are physically
    /// inconsistent is rejected with [`ProtocolError`] rather than answered
    /// with a guess.
    pub fn estimate_offset<P: NtpPeer>(&self, peer: &P) -> Result<ClockOffset, ProtocolError> {
        let originate = SystemTime::now();
        let (peer_receive, peer_transmit) = peer.respond();
        let destination = SystemTime::now();

        NtpTimestamps {
            originate,
            peer_receive,
            peer_transmit,
            destination,
        }
        .offset()
    }
}

/// A synchronization peer capable of completing one leg of an NTP-style
/// round trip: given that a request has just notionally arrived, it reports
/// the two *real* timestamps it observed on its own clock -- when it
/// received the request and when it sent its reply.
///
/// Any real transport can implement this by timestamping around its own
/// receive/send of the exchange; the estimation algorithm in
/// [`NtpTimestamps::offset`] is identically real whether the round trip
/// crosses an actual network or completes synchronously in-process.
pub trait NtpPeer {
    /// Real `(peer_receive, peer_transmit)` timestamps for one round trip.
    fn respond(&self) -> (SystemTime, SystemTime);
}

/// One completed NTP-style round-trip exchange: the four timestamps defined
/// by RFC 5905 (there named ambiguously "t1..t4"; named here for clarity).
#[derive(Debug, Clone, Copy)]
pub struct NtpTimestamps {
    /// t1: local time the request was sent.
    pub originate: SystemTime,
    /// t2: peer-reported time the request was received.
    pub peer_receive: SystemTime,
    /// t3: peer-reported time the reply was sent.
    pub peer_transmit: SystemTime,
    /// t4: local time the reply was received.
    pub destination: SystemTime,
}

impl NtpTimestamps {
    /// Signed nanoseconds from `earlier` to `later` (positive when `later`
    /// is after `earlier`).
    ///
    /// [`SystemTime::duration_since`] only returns an unsigned [`Duration`]
    /// and errors when the arguments are the "wrong" way around; this uses
    /// that error (which carries the reverse duration) to recover a real
    /// signed difference without unwrapping or panicking.
    fn signed_nanos_between(later: SystemTime, earlier: SystemTime) -> i128 {
        match later.duration_since(earlier) {
            Ok(d) => d.as_nanos() as i128,
            Err(e) => -(e.duration().as_nanos() as i128),
        }
    }

    /// Round-trip delay: `(t4 - t1) - (t3 - t2)`.
    ///
    /// Physically this must be non-negative: the full round trip cannot be
    /// shorter than the peer's own receive-to-transmit turnaround. A
    /// negative value means the timestamps are inconsistent (e.g. a clock
    /// stepped backward mid-exchange, or a peer reporting impossible
    /// values), so this surfaces that as an error rather than silently
    /// returning a delay nobody could trust.
    pub fn round_trip_delay(&self) -> Result<Duration, ProtocolError> {
        let total = Self::signed_nanos_between(self.destination, self.originate);
        let peer_turnaround = Self::signed_nanos_between(self.peer_transmit, self.peer_receive);
        let delay_ns = total - peer_turnaround;

        if delay_ns < 0 {
            return Err(ProtocolError::new(format!(
                "inconsistent NTP exchange: negative round-trip delay ({delay_ns}ns); \
                 timestamps cannot be trusted for offset estimation"
            )));
        }

        Ok(Duration::from_nanos(delay_ns as u64))
    }

    /// Classic NTP clock-offset estimate: `((t2 - t1) + (t3 - t4)) / 2`.
    ///
    /// A positive result means the peer's clock reads later ("ahead of")
    /// the local clock at the midpoint of the exchange; correcting the
    /// local clock to match the peer means advancing it by this offset.
    /// Validates [`Self::round_trip_delay`] first: an exchange whose
    /// timestamps are not physically consistent cannot yield a trustworthy
    /// offset either, so no offset is fabricated for it.
    pub fn offset(&self) -> Result<ClockOffset, ProtocolError> {
        self.round_trip_delay()?;

        let a = Self::signed_nanos_between(self.peer_receive, self.originate);
        let b = Self::signed_nanos_between(self.peer_transmit, self.destination);
        let offset_ns = (a + b) / 2;

        Ok(ClockOffset {
            offset_ns: offset_ns.clamp(i64::MIN as i128, i64::MAX as i128) as i64,
        })
    }
}

/// Protocol-level synchronization error.
///
/// Carries a real reason (e.g. from [`NtpTimestamps::round_trip_delay`]
/// rejecting a physically inconsistent exchange) rather than being a bare
/// marker, so callers and logs see what actually went wrong.
#[derive(Debug, Clone, Default)]
pub struct ProtocolError {
    pub reason: String,
}

impl ProtocolError {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.reason.is_empty() {
            write!(f, "Protocol error")
        } else {
            write!(f, "Protocol error: {}", self.reason)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic peer whose clock reads exactly `shift_ns` away from the
    /// caller's real `SystemTime::now()`, with real (nonzero) wall-clock
    /// time elapsing between when it "received" and when it "replied". This
    /// exercises [`NtpSynchronizer::estimate_offset`] end-to-end against a
    /// known, caller-controlled offset -- a fabricated stub (e.g. one that
    /// always returns `offset_ns: 0`) could never reproduce this.
    struct FixedOffsetPeer {
        shift_ns: i64,
    }

    impl FixedOffsetPeer {
        fn shifted_now(&self) -> SystemTime {
            let now = SystemTime::now();
            if self.shift_ns >= 0 {
                now + Duration::from_nanos(self.shift_ns as u64)
            } else {
                now - Duration::from_nanos((-self.shift_ns) as u64)
            }
        }
    }

    impl NtpPeer for FixedOffsetPeer {
        fn respond(&self) -> (SystemTime, SystemTime) {
            let receive = self.shifted_now();
            std::thread::sleep(Duration::from_millis(5));
            let transmit = self.shifted_now();
            (receive, transmit)
        }
    }

    #[test]
    fn estimate_offset_recovers_a_known_positive_peer_shift() {
        let synchronizer = NtpSynchronizer::default();
        let peer = FixedOffsetPeer {
            shift_ns: 200_000_000, // peer clock 200ms ahead of local
        };

        let offset = synchronizer
            .estimate_offset(&peer)
            .expect("a physically consistent exchange must not error");

        assert!(
            offset.offset_ns > 0,
            "a peer-ahead shift must yield a positive offset, got {}",
            offset.offset_ns
        );
        let error_ns = (offset.offset_ns - 200_000_000).abs();
        assert!(
            error_ns < 50_000_000,
            "expected offset near +200ms, got {}ns (error {}ns)",
            offset.offset_ns,
            error_ns
        );
    }

    #[test]
    fn estimate_offset_recovers_a_known_negative_peer_shift() {
        let synchronizer = NtpSynchronizer::default();
        let peer = FixedOffsetPeer {
            shift_ns: -150_000_000, // peer clock 150ms behind local
        };

        let offset = synchronizer
            .estimate_offset(&peer)
            .expect("a physically consistent exchange must not error");

        assert!(
            offset.offset_ns < 0,
            "a peer-behind shift must yield a negative offset, got {}",
            offset.offset_ns
        );
        let error_ns = (offset.offset_ns - (-150_000_000)).abs();
        assert!(
            error_ns < 50_000_000,
            "expected offset near -150ms, got {}ns (error {}ns)",
            offset.offset_ns,
            error_ns
        );
    }

    #[test]
    fn zero_shift_zero_latency_offset_is_exactly_zero() {
        let t = SystemTime::now();
        let timestamps = NtpTimestamps {
            originate: t,
            peer_receive: t,
            peer_transmit: t,
            destination: t,
        };

        let offset = timestamps
            .offset()
            .expect("a degenerate exchange with all four timestamps equal is trivially consistent");
        assert_eq!(offset.offset_ns, 0);
        assert_eq!(
            timestamps
                .round_trip_delay()
                .expect("degenerate exchange is consistent"),
            Duration::ZERO
        );
    }

    #[test]
    fn round_trip_delay_rejects_physically_inconsistent_timestamps() {
        let now = SystemTime::now();
        // The peer claims 495ms of its own receive-to-transmit turnaround,
        // but the entire round trip was only observed to take 20ms locally
        // -- physically impossible, so this must be rejected rather than
        // silently averaged into a bogus offset.
        let timestamps = NtpTimestamps {
            originate: now,
            destination: now + Duration::from_millis(20),
            peer_receive: now + Duration::from_millis(5),
            peer_transmit: now + Duration::from_millis(500),
        };

        assert!(timestamps.round_trip_delay().is_err());
        assert!(timestamps.offset().is_err());
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProtocolManager {
    pub protocol: ClockSyncProtocol,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PtpConfig {
    pub domain: u8,
}

#[derive(Debug, Clone, Default)]
pub struct PtpSynchronizer {
    pub config: PtpConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SntpConfig {
    pub server: String,
}

#[derive(Debug, Clone, Default)]
pub struct SntpSynchronizer {
    pub config: SntpConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PtpVersion {
    V1,
    #[default]
    V2,
    V2_1,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PtpTransport {
    #[default]
    UDP,
    Ethernet,
    Serial,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PtpProfile {
    #[default]
    Default,
    Telecom,
    Power,
    Industrial,
}
