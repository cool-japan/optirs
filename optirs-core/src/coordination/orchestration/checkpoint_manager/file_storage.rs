//! Filesystem-backed [`CheckpointStorage`] implementation.
//!
//! [`InMemoryCheckpointStorage`] is
//! [`CheckpointManager`]'s batteries-included default, but it does not
//! survive process restarts: everything lives in a `HashMap`. This module
//! adds the persistence half of the F9 fix -- a real, on-disk implementor
//! of [`CheckpointStorage`] that writes one file per checkpoint under a
//! configured base directory.
//!
//! ## On-disk format
//!
//! ```text
//! [MAGIC: 4 bytes "OCK1"][FORMAT_VERSION: u16 LE]
//! [oxicode checksum-wrapped payload: MAGIC(3) VERSION(1) LEN(8) CRC32(4) DATA]
//! ```
//!
//! The outer `MAGIC`/`FORMAT_VERSION` pair is this module's own -- bumping
//! `FILE_FORMAT_VERSION` lets a future version detect and honestly reject
//! (rather than misdecode) a file written by an incompatible schema,
//! before ever touching the inner payload. The inner payload is produced
//! by [`oxicode::serde::encode_to_vec`] (oxicode is the COOLJAPAN pure-Rust
//! `bincode` replacement; `bincode` itself is banned by `deny.toml`) and
//! wrapped with [`oxicode::checksum::wrap_with_checksum`], which prepends
//! its own magic/version/length/CRC32 header. Corruption or tampering
//! anywhere in the payload is caught by the CRC32 check before a single
//! byte is handed to the `Deserialize` impl.
//!
//! ## Atomicity
//!
//! [`FileCheckpointStorage::store`] never writes to the final
//! `<checkpoint_id>.ckpt` path directly. It writes to a uniquely-named
//! `*.tmp` sibling file, `fsync`s it, and only then [`std::fs::rename`]s it
//! into place -- atomic on the same filesystem on every target this crate
//! supports. A process that dies mid-write leaves the `.tmp` file behind,
//! never a half-written `.ckpt` file; [`FileCheckpointStorage::list`],
//! [`FileCheckpointStorage::exists`], and
//! [`FileCheckpointStorage::get_statistics`] all filter strictly by the
//! `.ckpt` extension, so a stray `.tmp` file is simply invisible to them,
//! regardless of how it was left behind.

use crate::error::{OptimError, Result};
use scirs2_core::numeric::Float;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::Debug;
use std::fs;
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::functions::CheckpointStorage;
use super::types::{
    Checkpoint, CheckpointManager, InMemoryCheckpointStorage, StorageConfig, StorageStatistics,
};
use super::types_15::{CheckpointConfiguration, CheckpointMetadata};

/// Magic bytes identifying an OptiRS `FileCheckpointStorage` file.
/// Deliberately distinct from oxicode's own checksum-wrapper magic
/// (`b"OXH"`) so a foreign or badly truncated file is rejected at the
/// very first check, before any decode work is attempted.
const FILE_MAGIC: [u8; 4] = *b"OCK1";

/// On-disk schema version for the `Checkpoint<T>` payload this module
/// writes. Bump this when a change to `CheckpointData<T>`'s field graph
/// would break binary compatibility with files written by an older
/// version of this module, and extend [`decode_checkpoint`] to branch on
/// the version it reads if reading older files must keep working.
const FILE_FORMAT_VERSION: u16 = 1;

/// `FILE_MAGIC` (4 bytes) + `FILE_FORMAT_VERSION` (2 bytes, little-endian).
const FILE_HEADER_LEN: usize = FILE_MAGIC.len() + 2;

/// Extension used for a fully-written, durable checkpoint file. Anything
/// without this exact extension (in particular, this module's own `*.tmp`
/// temp files) is invisible to [`FileCheckpointStorage::list`] and
/// [`FileCheckpointStorage::get_statistics`].
const CHECKPOINT_EXTENSION: &str = "ckpt";

/// Serializes `checkpoint` and wraps it with this module's versioned
/// header plus oxicode's CRC32 checksum. See the module docs for the exact
/// byte layout.
///
/// # Errors
/// Returns [`OptimError::InvalidConfig`] if oxicode's serde-compatible
/// encoder rejects `checkpoint` (see `oxicode::serde`'s documented
/// unsupported-attribute list -- none of which this crate's checkpoint
/// types use).
fn encode_checkpoint<T: Float + Debug + Send + Sync + 'static + Serialize>(
    checkpoint: &Checkpoint<T>,
) -> Result<Vec<u8>> {
    let payload =
        oxicode::serde::encode_to_vec(checkpoint, oxicode::config::standard()).map_err(|e| {
            OptimError::InvalidConfig(format!("failed to encode checkpoint for storage: {e}"))
        })?;
    let checksummed = oxicode::checksum::wrap_with_checksum(&payload);

    let mut out = Vec::with_capacity(FILE_HEADER_LEN + checksummed.len());
    out.extend_from_slice(&FILE_MAGIC);
    out.extend_from_slice(&FILE_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&checksummed);
    Ok(out)
}

/// Inverse of [`encode_checkpoint`]. Fails honestly -- never panics, never
/// falls back to a default/empty checkpoint -- at whichever stage first
/// detects a problem: truncation, wrong magic, unsupported format version,
/// CRC32 mismatch (corruption or tampering), or a malformed payload.
fn decode_checkpoint<T: Float + Debug + Send + Sync + 'static + DeserializeOwned>(
    bytes: &[u8],
    checkpoint_id: &str,
) -> Result<Checkpoint<T>> {
    if bytes.len() < FILE_HEADER_LEN {
        return Err(OptimError::InvalidConfig(format!(
            "checkpoint '{checkpoint_id}' file is truncated: {} byte(s), expected at least a {FILE_HEADER_LEN}-byte header",
            bytes.len()
        )));
    }
    let (magic, rest) = bytes.split_at(FILE_MAGIC.len());
    if magic != FILE_MAGIC {
        return Err(OptimError::InvalidConfig(format!(
            "checkpoint '{checkpoint_id}' file has an invalid header (not a FileCheckpointStorage file, or the header itself is corrupted)"
        )));
    }
    let (version_bytes, rest) = rest.split_at(2);
    let version = u16::from_le_bytes([version_bytes[0], version_bytes[1]]);
    if version != FILE_FORMAT_VERSION {
        return Err(OptimError::InvalidConfig(format!(
            "checkpoint '{checkpoint_id}' file format version {version} is not supported by this build (expected {FILE_FORMAT_VERSION})"
        )));
    }

    let payload = oxicode::checksum::verify_checksum(rest).map_err(|e| {
        OptimError::InvalidConfig(format!(
            "checkpoint '{checkpoint_id}' failed integrity verification (file is corrupted or was tampered with): {e}"
        ))
    })?;

    let (checkpoint, _consumed) = oxicode::serde::decode_owned_from_slice::<Checkpoint<T>, _>(
        payload,
        oxicode::config::standard(),
    )
    .map_err(|e| {
        OptimError::InvalidConfig(format!(
            "failed to decode checkpoint '{checkpoint_id}' payload: {e}"
        ))
    })?;
    Ok(checkpoint)
}

/// A [`CheckpointStorage`] backend that persists each checkpoint as one
/// file on the local filesystem, surviving process restarts (unlike
/// [`InMemoryCheckpointStorage`], this module's sibling batteries-included
/// backend, which does not). See the module docs for the on-disk format,
/// atomicity, and corruption-detection guarantees.
#[derive(Debug, Clone)]
pub struct FileCheckpointStorage<T> {
    base_dir: PathBuf,
    _marker: PhantomData<T>,
}

impl<T> FileCheckpointStorage<T> {
    /// Opens `base_dir` as the on-disk root for this backend, creating it
    /// (and any missing parent directories) if it does not already exist.
    /// Each checkpoint is stored as `base_dir/<checkpoint_id>.ckpt`.
    ///
    /// # Errors
    /// Returns [`OptimError::IO`] if `base_dir` cannot be created, or
    /// [`OptimError::InvalidConfig`] if the path exists but is not a
    /// directory.
    pub fn new(base_dir: impl Into<PathBuf>) -> Result<Self> {
        let base_dir = base_dir.into();
        fs::create_dir_all(&base_dir)?;
        if !base_dir.is_dir() {
            return Err(OptimError::InvalidConfig(format!(
                "FileCheckpointStorage base path {} exists but is not a directory",
                base_dir.display()
            )));
        }
        Ok(Self {
            base_dir,
            _marker: PhantomData,
        })
    }

    /// The base directory this backend was opened with.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Rejects a `checkpoint_id` that would be unsafe to embed directly
    /// into a file name: empty, `.`/`..`, or containing a path separator
    /// or NUL byte. Without this, a caller-controlled ID could otherwise
    /// escape `base_dir` or collide with this module's own temp-file
    /// naming convention.
    fn validate_checkpoint_id(checkpoint_id: &str) -> Result<()> {
        if checkpoint_id.is_empty()
            || checkpoint_id == "."
            || checkpoint_id == ".."
            || checkpoint_id.contains('/')
            || checkpoint_id.contains('\\')
            || checkpoint_id.contains('\0')
        {
            return Err(OptimError::InvalidConfig(format!(
                "invalid checkpoint_id {checkpoint_id:?}: must be non-empty and free of path separators"
            )));
        }
        Ok(())
    }

    fn checkpoint_path(&self, checkpoint_id: &str) -> Result<PathBuf> {
        Self::validate_checkpoint_id(checkpoint_id)?;
        Ok(self
            .base_dir
            .join(format!("{checkpoint_id}.{CHECKPOINT_EXTENSION}")))
    }

    /// A sibling path guaranteed not to collide with any other concurrent
    /// write to the same `checkpoint_id` (pid + nanosecond timestamp), and
    /// guaranteed to never end in `.{CHECKPOINT_EXTENSION}` (it ends in
    /// `.tmp`), so it can never be mistaken for a real checkpoint file by
    /// [`Self::list`]/[`Self::get_statistics`]'s extension filter -- no
    /// matter what `checkpoint_id` itself contains.
    fn temp_path(&self, checkpoint_id: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        self.base_dir.join(format!(
            "{checkpoint_id}.{}.{nanos}.tmp",
            std::process::id()
        ))
    }

    /// Writes `bytes` to a temp file, `fsync`s it, then renames it onto
    /// `final_path`. On any failure, best-effort removes the temp file
    /// (ignoring that cleanup's own result) and returns the *original*
    /// error -- cleanup never masks or replaces the real failure.
    fn write_atomically(&self, final_path: &Path, bytes: &[u8], checkpoint_id: &str) -> Result<()> {
        let tmp_path = self.temp_path(checkpoint_id);

        let write_result: Result<()> = (|| {
            let mut file = fs::File::create(&tmp_path)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            Ok(())
        })();
        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }

        if let Err(e) = fs::rename(&tmp_path, final_path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(OptimError::from(e));
        }
        Ok(())
    }

    /// Maps a filesystem [`std::io::Error`] to an honest, checkpoint-
    /// identifying [`OptimError`]: [`OptimError::InvalidConfig`] with a
    /// "not found" message for [`std::io::ErrorKind::NotFound`], the raw
    /// [`OptimError::IO`] for anything else (permissions, disk full, ...).
    fn not_found_or_io(checkpoint_id: &str, path: &Path, e: std::io::Error) -> OptimError {
        if e.kind() == std::io::ErrorKind::NotFound {
            OptimError::InvalidConfig(format!(
                "checkpoint '{checkpoint_id}' not found in file storage at {}",
                path.display()
            ))
        } else {
            OptimError::IO(e)
        }
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Serialize + DeserializeOwned> CheckpointStorage<T>
    for FileCheckpointStorage<T>
{
    fn store(&mut self, checkpoint: &Checkpoint<T>) -> Result<String> {
        let final_path = self.checkpoint_path(&checkpoint.checkpoint_id)?;
        let bytes = encode_checkpoint(checkpoint)?;
        self.write_atomically(&final_path, &bytes, &checkpoint.checkpoint_id)?;
        Ok(format!("file://{}", final_path.display()))
    }

    fn retrieve(&self, checkpoint_id: &str) -> Result<Checkpoint<T>> {
        let path = self.checkpoint_path(checkpoint_id)?;
        let bytes = fs::read(&path).map_err(|e| Self::not_found_or_io(checkpoint_id, &path, e))?;
        decode_checkpoint(&bytes, checkpoint_id)
    }

    fn delete(&mut self, checkpoint_id: &str) -> Result<()> {
        let path = self.checkpoint_path(checkpoint_id)?;
        fs::remove_file(&path).map_err(|e| Self::not_found_or_io(checkpoint_id, &path, e))
    }

    /// Lists checkpoint IDs by scanning `base_dir` for `*.ckpt` files.
    /// With `workflow_id: None`, IDs are read straight from file names (no
    /// I/O beyond the directory listing itself). With `workflow_id:
    /// Some(_)`, each candidate file must be fully decoded (magic,
    /// version, checksum, deserialize) to inspect its `workflow_id` field,
    /// so a corrupted checkpoint under `base_dir` fails the *whole* call
    /// rather than being silently skipped -- consistent with this crate's
    /// no-silent-failure policy elsewhere in the checkpoint manager.
    fn list(&self, workflow_id: Option<&str>) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.base_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some(CHECKPOINT_EXTENSION) {
                continue;
            }
            let Some(checkpoint_id) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            match workflow_id {
                None => ids.push(checkpoint_id.to_string()),
                Some(wf) => {
                    let checkpoint = decode_checkpoint::<T>(&fs::read(&path)?, checkpoint_id)?;
                    if checkpoint.workflow_id == wf {
                        ids.push(checkpoint_id.to_string());
                    }
                }
            }
        }
        Ok(ids)
    }

    fn exists(&self, checkpoint_id: &str) -> Result<bool> {
        let path = self.checkpoint_path(checkpoint_id)?;
        Ok(path.is_file())
    }

    fn get_metadata(&self, checkpoint_id: &str) -> Result<CheckpointMetadata<T>> {
        self.retrieve(checkpoint_id)
            .map(|checkpoint| checkpoint.metadata)
    }

    fn get_statistics(&self) -> Result<StorageStatistics> {
        let mut total_checkpoints = 0usize;
        let mut total_storage_bytes = 0usize;
        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(CHECKPOINT_EXTENSION) {
                continue;
            }
            total_checkpoints += 1;
            total_storage_bytes += entry.metadata()?.len() as usize;
        }
        let average_checkpoint_size = total_storage_bytes
            .checked_div(total_checkpoints)
            .unwrap_or(0);
        Ok(StorageStatistics {
            total_checkpoints,
            total_storage_bytes,
            average_checkpoint_size,
            // Real free-disk-space querying needs a platform syscall not
            // reachable through std alone; reporting a fabricated number
            // would be worse than reporting "unknown", matching the same
            // honest convention InMemoryCheckpointStorage::get_statistics
            // already uses for the fields it likewise cannot compute.
            utilization_percentage: 0.0,
            available_storage_bytes: usize::MAX,
        })
    }
}

/// Constructs a [`CheckpointStorage`] backend selected by
/// `config.backend_type`, for config-driven backend selection instead of
/// a caller having to construct one by hand:
///
/// - `"memory"` -- [`InMemoryCheckpointStorage`] (`config.location` is
///   ignored).
/// - `"file"` -- [`FileCheckpointStorage`], rooted at `config.location`
///   (created if it does not exist).
///
/// # Errors
/// Returns [`OptimError::InvalidConfig`] for any other `backend_type`, or
/// for `"file"` with an empty `location`.
pub fn storage_backend_from_config<T>(
    config: &StorageConfig,
) -> Result<Box<dyn CheckpointStorage<T>>>
where
    T: Float + Debug + Send + Sync + 'static + Clone + Serialize + DeserializeOwned,
{
    match config.backend_type.as_str() {
        "memory" => Ok(Box::new(InMemoryCheckpointStorage::<T>::new())),
        "file" => {
            if config.location.trim().is_empty() {
                return Err(OptimError::InvalidConfig(
                    "storage backend_type \"file\" requires a non-empty `location` (base directory path)"
                        .to_string(),
                ));
            }
            Ok(Box::new(FileCheckpointStorage::<T>::new(&config.location)?))
        }
        other => Err(OptimError::InvalidConfig(format!(
            "unsupported storage backend_type {other:?}; expected \"memory\" or \"file\""
        ))),
    }
}

impl<T: Float + Debug + Send + Sync + 'static + Default + Clone + Serialize + DeserializeOwned>
    CheckpointManager<T>
{
    /// Convenience constructor that derives the storage backend from
    /// `config.storage_config` via [`storage_backend_from_config`],
    /// instead of requiring the caller to construct and pass one
    /// explicitly the way [`CheckpointManager::new`] does. Prefer
    /// [`CheckpointManager::new`] directly when the desired backend isn't
    /// one `storage_backend_from_config` supports (e.g. a caller's own
    /// [`CheckpointStorage`] implementor).
    ///
    /// # Errors
    /// Propagates [`storage_backend_from_config`]'s errors for an
    /// unsupported or misconfigured `config.storage_config`.
    pub fn from_config(config: CheckpointConfiguration<T>) -> Result<Self> {
        let backend = storage_backend_from_config::<T>(&config.storage_config)?;
        Self::new(config, backend)
    }
}

#[cfg(test)]
mod tests {
    use super::super::functions::empty_checkpoint_data;
    use super::super::types::{
        CheckpointType, CompressorConfig, CreatorInfo, IndexerConfig, OptimizerState,
        RecoveryConfig,
    };
    use super::super::types_15::{
        AccessPermissions, CompressionInfo, OptimizerConfiguration, SchedulerConfig,
        ValidationStatus, ValidatorConfig,
    };
    use super::*;
    use scirs2_core::ndarray::Array1;
    use std::collections::HashMap;
    use std::time::Duration;

    /// Every test gets its own directory under `std::env::temp_dir()`,
    /// named from the test name + pid + a nanosecond timestamp, so nextest
    /// running these in parallel (separate processes, shared filesystem)
    /// never lets two tests' `list()`/`get_statistics()` assertions race
    /// against each other's files.
    fn unique_temp_dir(test_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "optirs-file-checkpoint-storage-{test_name}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create unique temp dir");
        dir
    }

    fn sample_checkpoint(id: &str, workflow_id: &str) -> Checkpoint<f64> {
        let mut parameters = HashMap::new();
        parameters.insert(
            "layer0.weight".to_string(),
            Array1::from(vec![1.5_f64, -2.25, 3.0, 0.0, 42.5]),
        );
        let optimizer_state = OptimizerState {
            optimizer_type: "adam".to_string(),
            parameters,
            buffers: HashMap::new(),
            configuration: OptimizerConfiguration {
                base_lr: 0.001,
                momentum: HashMap::new(),
                adaptive_params: HashMap::new(),
                regularization_params: HashMap::new(),
            },
            step_counter: 42,
            lr_history: vec![0.01, 0.005, 0.001],
        };
        let mut data = empty_checkpoint_data::<f64>();
        data.optimizer_state = Some(optimizer_state);

        Checkpoint {
            checkpoint_id: id.to_string(),
            workflow_id: workflow_id.to_string(),
            checkpoint_type: CheckpointType::Custom("nightly-eval".to_string()),
            data,
            metadata: CheckpointMetadata {
                description: "round-trip test checkpoint".to_string(),
                tags: vec!["test".to_string()],
                version: "1.0".to_string(),
                creator: CreatorInfo {
                    name: "file_storage_tests".to_string(),
                    email: None,
                    tool: "optirs-core".to_string(),
                    tool_version: "0.1.0".to_string(),
                },
                metrics: HashMap::new(),
                validation_status: ValidationStatus {
                    valid: true,
                    errors: Vec::new(),
                    warnings: Vec::new(),
                    validated_at: SystemTime::now(),
                    validator_version: "1.0".to_string(),
                },
                storage_location: String::new(),
                backup_locations: Vec::new(),
                permissions: AccessPermissions::default(),
            },
            created_at: SystemTime::now(),
            size_bytes: 128,
            hash: "test-hash".to_string(),
            compression: CompressionInfo {
                compression_time: Duration::from_millis(7),
                ..CompressionInfo::default()
            },
            dependencies: Vec::new(),
        }
    }

    // ---- round-trip save/load equality --------------------------------

    #[test]
    fn round_trip_preserves_array_enum_time_and_option_fields() {
        let dir = unique_temp_dir("round-trip");
        let mut storage = FileCheckpointStorage::<f64>::new(&dir).expect("new");
        let original = sample_checkpoint("ckpt-roundtrip", "wf-roundtrip");

        storage.store(&original).expect("store");
        let loaded = storage.retrieve("ckpt-roundtrip").expect("retrieve");

        assert_eq!(loaded.checkpoint_id, original.checkpoint_id);
        assert_eq!(loaded.workflow_id, original.workflow_id);
        // Payload-carrying enum variant: exercises the non-self-describing
        // tag encoding, not just a unit variant.
        assert_eq!(loaded.checkpoint_type, original.checkpoint_type);
        assert_eq!(
            loaded.created_at.duration_since(UNIX_EPOCH).unwrap(),
            original.created_at.duration_since(UNIX_EPOCH).unwrap(),
            "SystemTime must round-trip exactly"
        );
        assert_eq!(
            loaded.compression.compression_time, original.compression.compression_time,
            "Duration must round-trip exactly"
        );

        // The actual deliverable: a real Array1<T> through the codec.
        let original_weights = original
            .data
            .optimizer_state
            .as_ref()
            .expect("original has optimizer_state")
            .parameters
            .get("layer0.weight")
            .expect("original has layer0.weight");
        let loaded_weights = loaded
            .data
            .optimizer_state
            .as_ref()
            .expect("loaded must have optimizer_state")
            .parameters
            .get("layer0.weight")
            .expect("loaded must have layer0.weight");
        assert_eq!(loaded_weights.len(), original_weights.len());
        assert_eq!(loaded_weights.as_slice(), original_weights.as_slice());
        assert_eq!(
            loaded.data.optimizer_state.as_ref().unwrap().step_counter,
            42
        );

        // CheckpointData::model_state was never set: None must stay None.
        assert!(loaded.data.model_state.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- tamper detection ----------------------------------------------

    #[test]
    fn tampering_with_the_payload_is_detected_on_load() {
        let dir = unique_temp_dir("tamper");
        let mut storage = FileCheckpointStorage::<f64>::new(&dir).expect("new");
        let checkpoint = sample_checkpoint("ckpt-tamper", "wf-tamper");
        storage.store(&checkpoint).expect("store");

        // Must load cleanly *before* tampering -- otherwise a `retrieve`
        // that always errors would make this test pass for the wrong
        // reason.
        storage
            .retrieve("ckpt-tamper")
            .expect("must load successfully before tampering");

        let path = dir.join("ckpt-tamper.ckpt");
        let mut bytes = std::fs::read(&path).expect("read written file");
        // Flip a byte well inside the payload region (past our 6-byte
        // header and oxicode's 16-byte checksum header), so the mutation
        // lands in the actual checkpoint data, not incidentally in a
        // length/version field that might independently reject the file.
        let flip_at = bytes.len() - 3;
        bytes[flip_at] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write tampered bytes");

        let result = storage.retrieve("ckpt-tamper");
        assert!(
            result.is_err(),
            "a tampered checkpoint file must fail to load, not silently decode"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- partial-write recovery -----------------------------------------

    #[test]
    fn stray_temp_file_is_ignored_and_real_checkpoint_still_works() {
        let dir = unique_temp_dir("partial-write");
        let mut storage = FileCheckpointStorage::<f64>::new(&dir).expect("new");
        let checkpoint = sample_checkpoint("ckpt-real", "wf-partial");
        storage.store(&checkpoint).expect("store");

        // Simulate a crash mid-write: a `.tmp` file left behind by some
        // earlier, never-completed `store` call. `store`'s own real
        // temp-file naming always embeds pid+nanos, so this hand-written
        // name is guaranteed distinguishable from (and never collides
        // with) a real in-flight write.
        let stray_tmp = dir.join("ckpt-crashed.999999.123456789.tmp");
        std::fs::write(&stray_tmp, b"not a valid checkpoint file at all").expect("write stray tmp");

        let ids = storage.list(None).expect("list");
        assert!(
            ids.contains(&"ckpt-real".to_string()),
            "the real checkpoint must still be listed: {ids:?}"
        );
        assert!(
            !ids.iter().any(|id| id.contains("ckpt-crashed")),
            "the stray temp file must not appear in list(): {ids:?}"
        );

        // The real checkpoint must still retrieve correctly alongside the
        // ignored stray file.
        let loaded = storage
            .retrieve("ckpt-real")
            .expect("retrieve real checkpoint");
        assert_eq!(loaded.checkpoint_id, "ckpt-real");

        let stats = storage.get_statistics().expect("get_statistics");
        assert_eq!(
            stats.total_checkpoints, 1,
            "the stray .tmp file must not be counted"
        );

        // The stray file itself is untouched (list/get_statistics ignore
        // it, but don't delete it out from under a process that might
        // still be writing it).
        assert!(stray_tmp.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- delete / exists / not-found honesty ----------------------------

    #[test]
    fn delete_and_missing_lookups_fail_honestly() {
        let dir = unique_temp_dir("not-found");
        let mut storage = FileCheckpointStorage::<f64>::new(&dir).expect("new");

        assert!(!storage.exists("does-not-exist").expect("exists"));
        assert!(storage.retrieve("does-not-exist").is_err());
        assert!(storage.delete("does-not-exist").is_err());

        let checkpoint = sample_checkpoint("ckpt-del", "wf-del");
        storage.store(&checkpoint).expect("store");
        assert!(storage.exists("ckpt-del").expect("exists"));
        storage.delete("ckpt-del").expect("delete");
        assert!(!storage.exists("ckpt-del").expect("exists"));
        assert!(storage.retrieve("ckpt-del").is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn list_filters_by_workflow_id() {
        let dir = unique_temp_dir("list-filter");
        let mut storage = FileCheckpointStorage::<f64>::new(&dir).expect("new");
        storage
            .store(&sample_checkpoint("a", "wf-1"))
            .expect("store a");
        storage
            .store(&sample_checkpoint("b", "wf-2"))
            .expect("store b");

        let mut wf1 = storage.list(Some("wf-1")).expect("list wf-1");
        wf1.sort();
        assert_eq!(wf1, vec!["a".to_string()]);

        let mut all = storage.list(None).expect("list all");
        all.sort();
        assert_eq!(all, vec!["a".to_string(), "b".to_string()]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn invalid_checkpoint_id_is_rejected_honestly() {
        let dir = unique_temp_dir("invalid-id");
        let mut storage = FileCheckpointStorage::<f64>::new(&dir).expect("new");
        let mut checkpoint = sample_checkpoint("../escape", "wf-x");
        checkpoint.checkpoint_id = "../escape".to_string();

        assert!(
            storage.store(&checkpoint).is_err(),
            "a path-traversal-shaped checkpoint_id must be rejected, not silently escape base_dir"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- config-driven backend selection --------------------------------

    #[test]
    fn storage_backend_from_config_selects_file_backend() {
        let dir = unique_temp_dir("factory-file");
        let config = StorageConfig {
            backend_type: "file".to_string(),
            location: dir.to_string_lossy().to_string(),
            credentials: None,
            options: HashMap::new(),
        };

        let mut backend = storage_backend_from_config::<f64>(&config).expect("factory");
        let checkpoint = sample_checkpoint("ckpt-factory", "wf-factory");
        backend
            .store(&checkpoint)
            .expect("store via factory backend");
        assert!(dir.join("ckpt-factory.ckpt").is_file());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn storage_backend_from_config_rejects_unknown_backend_type() {
        let config = StorageConfig {
            backend_type: "quantum-foam".to_string(),
            location: String::new(),
            credentials: None,
            options: HashMap::new(),
        };
        assert!(storage_backend_from_config::<f64>(&config).is_err());
    }

    #[test]
    fn checkpoint_manager_from_config_round_trips_through_a_file_backend() {
        let dir = unique_temp_dir("manager-from-config");
        let config = CheckpointConfiguration {
            default_checkpoint_type: CheckpointType::Full,
            storage_config: StorageConfig {
                backend_type: "file".to_string(),
                location: dir.to_string_lossy().to_string(),
                credentials: None,
                options: HashMap::new(),
            },
            compression_config: CompressorConfig {
                enable_compression: false,
                default_compression_level: 0,
                size_threshold: usize::MAX,
                custom_params: HashMap::new(),
            },
            validation_config: ValidatorConfig {
                strict_validation: false,
                validation_timeout: Duration::from_secs(5),
                custom_params: HashMap::new(),
            },
            recovery_config: RecoveryConfig {
                default_timeout: Duration::from_secs(30),
                max_attempts: 3,
                retry_delay: Duration::from_secs(1),
                custom_params: HashMap::new(),
            },
            scheduling_config: SchedulerConfig {
                default_interval: Duration::from_secs(60),
                max_frequency: 1.0,
                min_interval: Duration::from_secs(1),
                adaptive_params: HashMap::new(),
            },
            indexing_config: IndexerConfig {
                rebuild_interval: Duration::from_secs(300),
                compaction_threshold: 0.5,
                enable_caching: true,
                custom_params: HashMap::new(),
            },
        };

        let mut manager =
            CheckpointManager::<f64>::from_config(config).expect("CheckpointManager::from_config");

        let mut data = empty_checkpoint_data::<f64>();
        data.custom_state
            .insert("round_trip".to_string(), vec![9, 8, 7]);

        let checkpoint_id = manager
            .create_checkpoint("wf-1".to_string(), CheckpointType::Full, data)
            .expect("create_checkpoint");

        // Survives a brand-new manager instance pointed at the same
        // directory -- i.e. actually persisted, not just held in memory.
        drop(manager);
        let reopened_config_backend = FileCheckpointStorage::<f64>::new(&dir).expect("reopen");
        let reloaded = reopened_config_backend
            .retrieve(&checkpoint_id)
            .expect("retrieve after reopening backend fresh");
        assert_eq!(
            reloaded.data.custom_state.get("round_trip"),
            Some(&vec![9, 8, 7])
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
