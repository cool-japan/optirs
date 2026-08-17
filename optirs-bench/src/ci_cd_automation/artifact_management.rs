// Artifact Management and Storage
//
// This module provides comprehensive artifact management capabilities for CI/CD automation,
// including multiple storage providers, upload/download functionality, retention policies,
// and artifact lifecycle management.

use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::config::{
    ArtifactDownloadConfig, ArtifactRetentionPolicy, ArtifactStorageConfig,
    ArtifactStorageProvider, ArtifactUploadConfig, CompressionAlgorithm, EncryptionConfig,
    ParallelUploadConfig,
};

/// Artifact manager for handling storage and retrieval
#[derive(Debug)]
pub struct ArtifactManager {
    /// Storage provider implementation
    pub storage_provider: Box<dyn ArtifactStorage>,
    /// Configuration settings
    pub config: ArtifactStorageConfig,
    /// Artifact registry for tracking
    pub registry: ArtifactRegistry,
    /// Upload manager
    pub upload_manager: UploadManager,
    /// Download manager
    pub download_manager: DownloadManager,
    /// Retention manager
    pub retention_manager: RetentionManager,
}

/// Trait for artifact storage implementations
pub trait ArtifactStorage: std::fmt::Debug + Send + Sync {
    /// Upload an artifact to storage
    fn upload(&self, local_path: &Path, remote_key: &str) -> Result<String>;

    /// Download an artifact from storage
    fn download(&self, remote_key: &str, local_path: &Path) -> Result<()>;

    /// Delete an artifact from storage
    fn delete(&self, remote_key: &str) -> Result<()>;

    /// List artifacts with optional prefix filter
    fn list(&self, prefix: Option<&str>) -> Result<Vec<ArtifactInfo>>;

    /// Check if an artifact exists
    fn exists(&self, remote_key: &str) -> Result<bool>;

    /// Get metadata for an artifact
    fn get_metadata(&self, remote_key: &str) -> Result<ArtifactMetadata>;

    /// Get storage statistics
    fn get_storage_stats(&self) -> Result<StorageStatistics>;

    /// Validate connection and permissions
    fn validate_connection(&self) -> Result<()>;
}

/// Artifact registry for tracking uploaded artifacts
#[derive(Debug, Clone)]
pub struct ArtifactRegistry {
    /// Registry of tracked artifacts
    pub artifacts: HashMap<String, ArtifactRecord>,
    /// Registry metadata
    pub metadata: RegistryMetadata,
}

/// Individual artifact record in the registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    /// Unique artifact ID
    pub id: String,
    /// Artifact name/key
    pub key: String,
    /// Local file path
    pub local_path: PathBuf,
    /// Remote storage path/key
    pub remote_key: String,
    /// Artifact metadata
    pub metadata: ArtifactMetadata,
    /// Upload timestamp
    pub uploaded_at: SystemTime,
    /// Last accessed timestamp
    pub last_accessed: Option<SystemTime>,
    /// Artifact tags
    pub tags: Vec<String>,
    /// Retention policy applied
    pub retention_policy: String,
    /// Upload status
    pub status: ArtifactStatus,
}

/// Artifact status in the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArtifactStatus {
    /// Pending upload
    Pending,
    /// Currently uploading
    Uploading,
    /// Successfully uploaded
    Uploaded,
    /// Upload failed
    Failed,
    /// Marked for deletion
    MarkedForDeletion,
    /// Deleted
    Deleted,
    /// Archived
    Archived,
}

/// Registry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    /// Registry version
    pub version: String,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last updated timestamp
    pub updated_at: SystemTime,
    /// Total artifacts tracked
    pub total_artifacts: usize,
    /// Registry statistics
    pub statistics: RegistryStatistics,
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStatistics {
    /// Total storage used (bytes)
    pub total_size_bytes: u64,
    /// Number of active artifacts
    pub active_artifacts: usize,
    /// Number of failed uploads
    pub failed_uploads: usize,
    /// Number of deleted artifacts
    pub deleted_artifacts: usize,
    /// Average artifact size
    pub average_size_bytes: f64,
}

/// Artifact information from storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactInfo {
    /// Artifact key/name
    pub key: String,
    /// Artifact size in bytes
    pub size: u64,
    /// Last modified timestamp
    pub last_modified: SystemTime,
    /// Content type/MIME type
    pub content_type: Option<String>,
    /// Storage class
    pub storage_class: Option<String>,
    /// ETag or checksum
    pub etag: Option<String>,
}

/// Artifact metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    /// File name
    pub filename: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Content type
    pub content_type: String,
    /// Checksum/hash
    pub checksum: String,
    /// Checksum algorithm
    pub checksum_algorithm: ChecksumAlgorithm,
    /// Compression information
    pub compression: Option<CompressionInfo>,
    /// Encryption information
    pub encryption: Option<EncryptionInfo>,
    /// Custom metadata fields
    pub custom_metadata: HashMap<String, String>,
}

/// Checksum algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChecksumAlgorithm {
    /// MD5 hash
    MD5,
    /// SHA-1 hash
    SHA1,
    /// SHA-256 hash
    SHA256,
    /// SHA-512 hash
    SHA512,
    /// CRC32 checksum
    CRC32,
    /// Blake2b hash
    Blake2b,
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Compression algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub level: u8,
    /// Original size before compression
    pub original_size: u64,
    /// Compressed size
    pub compressed_size: u64,
    /// Compression ratio
    pub compression_ratio: f64,
}

/// Encryption information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionInfo {
    /// Encryption algorithm used
    pub algorithm: String,
    /// Key identifier
    pub key_id: String,
    /// Initialization vector
    pub iv: Option<String>,
    /// Encryption metadata
    pub metadata: HashMap<String, String>,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStatistics {
    /// Total storage used (bytes)
    pub total_size_bytes: u64,
    /// Number of objects stored
    pub object_count: usize,
    /// Available storage space (bytes)
    pub available_space_bytes: Option<u64>,
    /// Storage utilization percentage
    pub utilization_percent: Option<f64>,
    /// Average object size
    pub average_object_size: f64,
}

/// Upload manager for handling file uploads
#[derive(Debug, Clone)]
pub struct UploadManager {
    /// Upload configuration
    pub config: ArtifactUploadConfig,
    /// Active uploads
    pub active_uploads: HashMap<String, UploadTask>,
    /// Upload history
    pub upload_history: Vec<UploadResult>,
}

/// Individual upload task
#[derive(Debug, Clone)]
pub struct UploadTask {
    /// Task ID
    pub id: String,
    /// Local file path
    pub local_path: PathBuf,
    /// Remote key
    pub remote_key: String,
    /// Upload progress
    pub progress: UploadProgress,
    /// Task status
    pub status: UploadStatus,
    /// Start timestamp
    pub started_at: SystemTime,
    /// Task configuration
    pub config: UploadTaskConfig,
}

/// Upload progress tracking
#[derive(Debug, Clone)]
pub struct UploadProgress {
    /// Bytes uploaded
    pub bytes_uploaded: u64,
    /// Total bytes to upload
    pub total_bytes: u64,
    /// Progress percentage (0.0 to 100.0)
    pub percentage: f64,
    /// Upload speed (bytes per second)
    pub speed_bps: f64,
    /// Estimated time remaining
    pub eta_seconds: Option<u64>,
}

/// Upload status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UploadStatus {
    /// Upload queued
    Queued,
    /// Upload in progress
    InProgress,
    /// Upload completed successfully
    Completed,
    /// Upload failed
    Failed,
    /// Upload cancelled
    Cancelled,
    /// Upload paused
    Paused,
}

/// Upload task configuration
#[derive(Debug, Clone)]
pub struct UploadTaskConfig {
    /// Enable compression
    pub compress: bool,
    /// Compression level
    pub compression_level: u8,
    /// Enable encryption
    pub encrypt: bool,
    /// Chunk size for multipart uploads
    pub chunk_size_bytes: usize,
    /// Number of retry attempts
    pub retry_attempts: u32,
}

/// Upload result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadResult {
    /// Task ID
    pub task_id: String,
    /// Remote key
    pub remote_key: String,
    /// Upload status
    pub status: UploadStatus,
    /// Total bytes uploaded
    pub bytes_uploaded: u64,
    /// Upload duration
    pub duration: Duration,
    /// Average upload speed
    pub average_speed_bps: f64,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Upload timestamp
    pub timestamp: SystemTime,
}

/// Download manager for handling file downloads
#[derive(Debug, Clone)]
pub struct DownloadManager {
    /// Download configuration
    pub config: ArtifactDownloadConfig,
    /// Download cache
    pub cache: DownloadCache,
    /// Active downloads
    pub active_downloads: HashMap<String, DownloadTask>,
}

/// Download cache management
#[derive(Debug, Clone)]
pub struct DownloadCache {
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Cache size limit (bytes)
    pub size_limit_bytes: u64,
    /// Current cache size (bytes)
    pub current_size_bytes: u64,
    /// Cache entries
    pub entries: HashMap<String, CacheEntry>,
    /// Cache statistics
    pub statistics: CacheStatistics,
}

/// Cache entry information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cache key
    pub key: String,
    /// Local file path
    pub local_path: PathBuf,
    /// File size
    pub size_bytes: u64,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last accessed timestamp
    pub last_accessed: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Cache hit score
    pub hit_score: f64,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Cache hit ratio
    pub hit_ratio: f64,
    /// Total cache evictions
    pub evictions: u64,
    /// Average file size
    pub average_file_size: f64,
}

/// Download task
#[derive(Debug, Clone)]
pub struct DownloadTask {
    /// Task ID
    pub id: String,
    /// Remote key
    pub remote_key: String,
    /// Local destination path
    pub local_path: PathBuf,
    /// Download progress
    pub progress: DownloadProgress,
    /// Task status
    pub status: DownloadStatus,
    /// Start timestamp
    pub started_at: SystemTime,
}

/// Download progress tracking
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Bytes downloaded
    pub bytes_downloaded: u64,
    /// Total bytes to download
    pub total_bytes: u64,
    /// Progress percentage (0.0 to 100.0)
    pub percentage: f64,
    /// Download speed (bytes per second)
    pub speed_bps: f64,
    /// Estimated time remaining
    pub eta_seconds: Option<u64>,
}

/// Download status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadStatus {
    /// Download queued
    Queued,
    /// Download in progress
    InProgress,
    /// Download completed successfully
    Completed,
    /// Download failed
    Failed,
    /// Download cancelled
    Cancelled,
    /// Served from cache
    FromCache,
}

/// Retention manager for artifact lifecycle
#[derive(Debug, Clone)]
pub struct RetentionManager {
    /// Retention policy
    pub policy: ArtifactRetentionPolicy,
    /// Cleanup scheduler
    pub scheduler: CleanupScheduler,
    /// Cleanup history
    pub cleanup_history: Vec<CleanupResult>,
}

/// Cleanup scheduler
#[derive(Debug, Clone)]
pub struct CleanupScheduler {
    /// Next cleanup time
    pub next_cleanup: SystemTime,
    /// Cleanup interval
    pub interval: Duration,
    /// Cleanup rules
    pub rules: Vec<CleanupRule>,
    /// Enabled status
    pub enabled: bool,
}

/// Cleanup rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: CleanupCondition,
    /// Action to take
    pub action: CleanupAction,
    /// Rule priority (higher number = higher priority)
    pub priority: u32,
    /// Enabled status
    pub enabled: bool,
}

/// Cleanup conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupCondition {
    /// Age-based condition
    Age { days: u32 },
    /// Size-based condition
    Size { max_size_gb: f64 },
    /// Count-based condition
    Count { max_count: usize },
    /// Tag-based condition
    Tag { tag: String },
    /// Status-based condition
    Status { status: ArtifactStatus },
    /// Custom condition
    Custom { expression: String },
}

/// Cleanup actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CleanupAction {
    /// Delete the artifact
    Delete,
    /// Archive the artifact
    Archive,
    /// Move to different storage class
    MoveToStorageClass { class: String },
    /// Compress the artifact
    Compress,
    /// Tag for review
    TagForReview,
    /// No action (just log)
    Log,
}

/// Cleanup operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupResult {
    /// Cleanup run ID
    pub run_id: String,
    /// Cleanup timestamp
    pub timestamp: SystemTime,
    /// Artifacts an enabled rule actually acted on (deleted or archived) --
    /// never counts an artifact that only *matched* a rule whose action is
    /// unimplemented or that failed a registry lookup.
    pub artifacts_processed: usize,
    /// Artifacts deleted
    pub artifacts_deleted: usize,
    /// Artifacts archived
    pub artifacts_archived: usize,
    /// Artifacts that matched an enabled rule but were left untouched,
    /// because the rule's action (`MoveToStorageClass`/`Compress`/
    /// `TagForReview`) has no implementation, or the matched key was no
    /// longer present in the registry by the time the action ran.
    pub artifacts_skipped: usize,
    /// Space freed (bytes)
    pub space_freed_bytes: u64,
    /// Cleanup duration
    pub duration: Duration,
    /// Cleanup summary
    pub summary: String,
}

/// Local filesystem storage implementation
#[derive(Debug, Clone)]
pub struct LocalArtifactStorage {
    /// Base storage path
    pub base_path: PathBuf,
    /// Storage configuration
    pub config: LocalStorageConfig,
}

/// Local storage configuration
#[derive(Debug, Clone)]
pub struct LocalStorageConfig {
    /// Create directories if they don't exist
    pub create_dirs: bool,
    /// File permissions for created files
    pub file_permissions: Option<u32>,
    /// Directory permissions for created directories
    pub dir_permissions: Option<u32>,
    /// Enable symbolic links
    pub allow_symlinks: bool,
}

/// Compute a real checksum of `buffer` for every [`ChecksumAlgorithm`] this
/// crate can implement without a new dependency. `Blake2b` is the one
/// exception (no Blake2 implementation is linked into this crate and
/// hand-rolling one is out of scope here); it returns an explicit `Err`
/// rather than a fabricated value shaped like `"checksum_<len>"`.
fn compute_checksum(buffer: &[u8], algorithm: ChecksumAlgorithm) -> Result<String> {
    match algorithm {
        ChecksumAlgorithm::SHA256 => {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(buffer);
            Ok(hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect())
        }
        ChecksumAlgorithm::SHA512 => {
            use sha2::{Digest, Sha512};
            let mut hasher = Sha512::new();
            hasher.update(buffer);
            Ok(hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect())
        }
        ChecksumAlgorithm::MD5 => {
            let mut hasher = md5::Context::new();
            hasher.consume(buffer);
            Ok(hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect())
        }
        ChecksumAlgorithm::SHA1 => Ok(sha1_hex(buffer)),
        ChecksumAlgorithm::CRC32 => Ok(format!("{:08x}", crc32(buffer))),
        ChecksumAlgorithm::Blake2b => Err(OptimError::UnsupportedOperation(
            "Blake2b checksums are not implemented (no Blake2 dependency is linked into this \
             crate); use SHA256, SHA512, SHA1, MD5, or CRC32 instead"
                .to_string(),
        )),
    }
}

/// Standard IEEE 802.3 CRC-32 (the algorithm used by zlib/PNG/gzip), computed
/// bit-by-bit with no lookup table so no extra static data is required.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// SHA-1 (FIPS 180-4). Used only as a checksum here, never for anything
/// security-sensitive; implemented directly since no `sha1` crate is a
/// workspace dependency and adding one for a single legacy checksum format
/// is not warranted.
fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x6745_2301;
    let mut h1: u32 = 0xEFCD_AB89;
    let mut h2: u32 = 0x98BA_DCFE;
    let mut h3: u32 = 0x1032_5476;
    let mut h4: u32 = 0xC3D2_E1F0;

    let message_bit_len = (data.len() as u64) * 8;
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&message_bit_len.to_be_bytes());

    for chunk in message.chunks(64) {
        let mut w = [0u32; 80];
        for (i, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for (i, word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDCu32),
                _ => (b ^ c ^ d, 0xCA62_C1D6u32),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut out = [0u8; 20];
    out[0..4].copy_from_slice(&h0.to_be_bytes());
    out[4..8].copy_from_slice(&h1.to_be_bytes());
    out[8..12].copy_from_slice(&h2.to_be_bytes());
    out[12..16].copy_from_slice(&h3.to_be_bytes());
    out[16..20].copy_from_slice(&h4.to_be_bytes());
    out
}

fn sha1_hex(data: &[u8]) -> String {
    sha1(data).iter().map(|b| format!("{b:02x}")).collect()
}

impl ArtifactManager {
    /// Create a new artifact manager
    pub fn new(config: ArtifactStorageConfig) -> Result<Self> {
        let storage_provider: Box<dyn ArtifactStorage> = match &config.provider {
            ArtifactStorageProvider::Local(path) => {
                Box::new(LocalArtifactStorage::new(path.clone()))
            }
            ArtifactStorageProvider::S3 {
                bucket,
                region,
                prefix,
            } => {
                return Err(OptimError::InvalidConfig(
                    "S3 storage not yet implemented".to_string(),
                ));
            }
            ArtifactStorageProvider::GCS { bucket, prefix } => {
                return Err(OptimError::InvalidConfig(
                    "GCS storage not yet implemented".to_string(),
                ));
            }
            ArtifactStorageProvider::AzureBlob {
                account,
                container,
                prefix,
            } => {
                return Err(OptimError::InvalidConfig(
                    "Azure Blob storage not yet implemented".to_string(),
                ));
            }
            ArtifactStorageProvider::FTP {
                host,
                port,
                path,
                secure,
            } => {
                return Err(OptimError::InvalidConfig(
                    "FTP storage not yet implemented".to_string(),
                ));
            }
            ArtifactStorageProvider::HTTP { base_url, auth } => {
                return Err(OptimError::InvalidConfig(
                    "HTTP storage not yet implemented".to_string(),
                ));
            }
        };

        Ok(Self {
            storage_provider,
            config: config.clone(),
            registry: ArtifactRegistry::new(),
            upload_manager: UploadManager::new(config.upload.clone()),
            download_manager: DownloadManager::new(config.download.clone()),
            retention_manager: RetentionManager::new(config.retention.clone()),
        })
    }

    /// Upload an artifact
    pub fn upload_artifact(
        &mut self,
        local_path: &Path,
        remote_key: &str,
        tags: Vec<String>,
    ) -> Result<String> {
        // Validate file exists
        if !local_path.exists() {
            return Err(OptimError::IO(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {:?}", local_path),
            )));
        }

        // `compress`/`encrypt` have no backend implementation in this crate
        // (compression would need an oxiarc-* dependency, encryption a
        // symmetric-cipher dependency; neither is linked in). Requesting
        // either must fail loudly rather than silently uploading the file
        // unmodified while `ArtifactMetadata::{compression,encryption}`
        // stays `None` -- a caller relying on "encrypt: true" to protect
        // data at rest must never get silent plaintext instead.
        if self.config.upload.compress {
            return Err(OptimError::UnsupportedOperation(
                "ArtifactUploadConfig::compress is set but no compression backend is \
                 implemented in this crate; set compress = false, or compress the file \
                 yourself before calling upload_artifact"
                    .to_string(),
            ));
        }
        if self.config.upload.encrypt {
            return Err(OptimError::UnsupportedOperation(
                "ArtifactUploadConfig::encrypt is set but no encryption backend is \
                 implemented in this crate; set encrypt = false, or encrypt the file \
                 yourself before calling upload_artifact"
                    .to_string(),
            ));
        }

        // Create artifact metadata
        let metadata = self.create_artifact_metadata(local_path)?;

        // Create upload task
        let task_id = self
            .upload_manager
            .create_upload_task(local_path, remote_key)?;

        // Perform upload
        let remote_url = self.storage_provider.upload(local_path, remote_key)?;

        // Record in registry
        let artifact_record = ArtifactRecord {
            id: uuid::Uuid::new_v4().to_string(),
            key: remote_key.to_string(),
            local_path: local_path.to_path_buf(),
            remote_key: remote_key.to_string(),
            metadata,
            uploaded_at: SystemTime::now(),
            last_accessed: None,
            tags,
            retention_policy: "default".to_string(),
            status: ArtifactStatus::Uploaded,
        };

        self.registry.add_artifact(artifact_record)?;

        // Update upload manager
        self.upload_manager.complete_upload(&task_id)?;

        Ok(remote_url)
    }

    /// Download an artifact
    pub fn download_artifact(&mut self, remote_key: &str, local_path: &Path) -> Result<()> {
        // Check cache first
        if let Some(cached_path) = self.download_manager.check_cache(remote_key)? {
            if cached_path != local_path {
                fs::copy(&cached_path, local_path).map_err(OptimError::IO)?;
            }
            return Ok(());
        }

        // Create download task
        let task_id = self
            .download_manager
            .create_download_task(remote_key, local_path)?;

        // Perform download
        self.storage_provider.download(remote_key, local_path)?;

        // Update cache
        self.download_manager.update_cache(remote_key, local_path)?;

        // Complete download task
        self.download_manager.complete_download(&task_id)?;

        Ok(())
    }

    /// Delete an artifact
    pub fn delete_artifact(&mut self, remote_key: &str) -> Result<()> {
        // Delete from storage
        self.storage_provider.delete(remote_key)?;

        // Update registry
        if let Some(artifact) = self.registry.get_artifact_mut(remote_key) {
            artifact.status = ArtifactStatus::Deleted;
        }

        // Remove from cache
        self.download_manager.remove_from_cache(remote_key)?;

        Ok(())
    }

    /// List artifacts
    pub fn list_artifacts(&self, prefix: Option<&str>) -> Result<Vec<ArtifactInfo>> {
        self.storage_provider.list(prefix)
    }

    /// Get artifact metadata
    pub fn get_artifact_metadata(&self, remote_key: &str) -> Result<ArtifactMetadata> {
        if let Some(artifact) = self.registry.get_artifact(remote_key) {
            Ok(artifact.metadata.clone())
        } else {
            self.storage_provider.get_metadata(remote_key)
        }
    }

    /// Run cleanup based on retention policies
    pub fn run_cleanup(&mut self) -> Result<CleanupResult> {
        self.retention_manager
            .run_cleanup(&mut self.registry, &*self.storage_provider)
    }

    /// Get storage statistics
    pub fn get_storage_statistics(&self) -> Result<StorageStatistics> {
        self.storage_provider.get_storage_stats()
    }

    /// Create artifact metadata from file
    fn create_artifact_metadata(&self, path: &Path) -> Result<ArtifactMetadata> {
        let file_size = fs::metadata(path)?.len();
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Compute checksum
        let checksum = self.compute_file_checksum(path, ChecksumAlgorithm::SHA256)?;

        // Determine content type
        let content_type = self.determine_content_type(path);

        Ok(ArtifactMetadata {
            filename,
            size_bytes: file_size,
            content_type,
            checksum,
            checksum_algorithm: ChecksumAlgorithm::SHA256,
            compression: None,
            encryption: None,
            custom_metadata: HashMap::new(),
        })
    }

    /// Compute file checksum
    fn compute_file_checksum(&self, path: &Path, algorithm: ChecksumAlgorithm) -> Result<String> {
        use std::io::Read;

        let mut file = fs::File::open(path).map_err(OptimError::IO)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(OptimError::IO)?;

        compute_checksum(&buffer, algorithm)
    }

    /// Determine content type from file extension
    fn determine_content_type(&self, path: &Path) -> String {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => "application/json".to_string(),
            Some("xml") => "application/xml".to_string(),
            Some("html") | Some("htm") => "text/html".to_string(),
            Some("txt") => "text/plain".to_string(),
            Some("pdf") => "application/pdf".to_string(),
            Some("zip") => "application/zip".to_string(),
            Some("tar") => "application/x-tar".to_string(),
            Some("gz") => "application/gzip".to_string(),
            _ => "application/octet-stream".to_string(),
        }
    }
}

impl LocalArtifactStorage {
    /// Create a new local storage instance
    pub fn new(base_path: PathBuf) -> Self {
        Self {
            base_path,
            config: LocalStorageConfig::default(),
        }
    }

    /// Create a new local storage instance with configuration
    pub fn new_with_config(base_path: PathBuf, config: LocalStorageConfig) -> Self {
        Self { base_path, config }
    }
}

impl ArtifactStorage for LocalArtifactStorage {
    fn upload(&self, local_path: &Path, remote_key: &str) -> Result<String> {
        let dest_path = self.base_path.join(remote_key);

        // Create parent directories if needed
        if let Some(parent) = dest_path.parent() {
            if self.config.create_dirs {
                fs::create_dir_all(parent).map_err(OptimError::IO)?;
            }
        }

        // Copy file
        fs::copy(local_path, &dest_path).map_err(OptimError::IO)?;

        // Set permissions if configured
        if let Some(permissions) = self.config.file_permissions {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&dest_path)?.permissions();
                perms.set_mode(permissions);
                fs::set_permissions(&dest_path, perms)?;
            }
        }

        Ok(dest_path.to_string_lossy().to_string())
    }

    fn download(&self, remote_key: &str, local_path: &Path) -> Result<()> {
        let source_path = self.base_path.join(remote_key);

        if !source_path.exists() {
            return Err(OptimError::IO(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Remote file not found: {}", remote_key),
            )));
        }

        // Create parent directories for local path
        if let Some(parent) = local_path.parent() {
            fs::create_dir_all(parent).map_err(OptimError::IO)?;
        }

        fs::copy(&source_path, local_path).map_err(OptimError::IO)?;

        Ok(())
    }

    fn delete(&self, remote_key: &str) -> Result<()> {
        let file_path = self.base_path.join(remote_key);

        if file_path.exists() {
            fs::remove_file(&file_path).map_err(OptimError::IO)?;
        }

        Ok(())
    }

    fn list(&self, prefix: Option<&str>) -> Result<Vec<ArtifactInfo>> {
        let mut artifacts = Vec::new();
        let search_path = if let Some(prefix) = prefix {
            self.base_path.join(prefix)
        } else {
            self.base_path.clone()
        };

        self.collect_artifacts(&search_path, &mut artifacts, prefix)?;
        Ok(artifacts)
    }

    fn exists(&self, remote_key: &str) -> Result<bool> {
        let file_path = self.base_path.join(remote_key);
        Ok(file_path.exists())
    }

    fn get_metadata(&self, remote_key: &str) -> Result<ArtifactMetadata> {
        use std::io::Read;

        let file_path = self.base_path.join(remote_key);
        let metadata = fs::metadata(&file_path).map_err(OptimError::IO)?;

        let filename = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut file = fs::File::open(&file_path).map_err(OptimError::IO)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(OptimError::IO)?;
        let checksum = compute_checksum(&buffer, ChecksumAlgorithm::SHA256)?;
        let content_type = match file_path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => "application/json".to_string(),
            Some("xml") => "application/xml".to_string(),
            Some("html") | Some("htm") => "text/html".to_string(),
            Some("txt") => "text/plain".to_string(),
            Some("zip") => "application/zip".to_string(),
            Some("tar") => "application/x-tar".to_string(),
            Some("gz") => "application/gzip".to_string(),
            _ => "application/octet-stream".to_string(),
        };

        Ok(ArtifactMetadata {
            filename,
            size_bytes: metadata.len(),
            content_type,
            checksum,
            checksum_algorithm: ChecksumAlgorithm::SHA256,
            compression: None,
            encryption: None,
            custom_metadata: HashMap::new(),
        })
    }

    fn get_storage_stats(&self) -> Result<StorageStatistics> {
        let mut total_size = 0u64;
        let mut object_count = 0usize;

        // Walk directory tree to calculate statistics
        fn walk_dir(dir: &Path, total_size: &mut u64, count: &mut usize) -> Result<()> {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    *total_size += fs::metadata(&path)?.len();
                    *count += 1;
                } else if path.is_dir() {
                    walk_dir(&path, total_size, count)?;
                }
            }
            Ok(())
        }

        if self.base_path.exists() && self.base_path.is_dir() {
            walk_dir(&self.base_path, &mut total_size, &mut object_count)?;
        }

        let average_object_size = if object_count > 0 {
            total_size as f64 / object_count as f64
        } else {
            0.0
        };

        Ok(StorageStatistics {
            total_size_bytes: total_size,
            object_count,
            available_space_bytes: None, // Could implement disk space check
            utilization_percent: None,
            average_object_size,
        })
    }

    fn validate_connection(&self) -> Result<()> {
        // For local storage, just check if base path exists and is accessible
        if !self.base_path.exists() {
            if self.config.create_dirs {
                fs::create_dir_all(&self.base_path).map_err(OptimError::IO)?;
            } else {
                return Err(OptimError::IO(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Base path does not exist: {:?}", self.base_path),
                )));
            }
        }

        // Test write access
        let test_file = self.base_path.join(".write_test");
        fs::write(&test_file, "test").map_err(OptimError::IO)?;
        fs::remove_file(&test_file).map_err(OptimError::IO)?;

        Ok(())
    }
}

impl LocalArtifactStorage {
    /// Recursively collect artifacts from directory
    fn collect_artifacts(
        &self,
        dir: &Path,
        artifacts: &mut Vec<ArtifactInfo>,
        prefix: Option<&str>,
    ) -> Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let relative_path = path.strip_prefix(&self.base_path).map_err(|e| {
                    OptimError::IO(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Failed to get relative path: {}", e),
                    ))
                })?;

                let key = relative_path.to_string_lossy().to_string();

                // Filter by prefix if specified
                if let Some(prefix) = prefix {
                    if !key.starts_with(prefix) {
                        continue;
                    }
                }

                let metadata = fs::metadata(&path)?;
                let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);

                artifacts.push(ArtifactInfo {
                    key,
                    size: metadata.len(),
                    last_modified: modified,
                    content_type: None,
                    storage_class: None,
                    etag: None,
                });
            } else if path.is_dir() {
                self.collect_artifacts(&path, artifacts, prefix)?;
            }
        }

        Ok(())
    }
}

// Implementation of supporting structures

impl Default for ArtifactRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ArtifactRegistry {
    /// Create a new artifact registry
    pub fn new() -> Self {
        Self {
            artifacts: HashMap::new(),
            metadata: RegistryMetadata {
                version: "1.0.0".to_string(),
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                total_artifacts: 0,
                statistics: RegistryStatistics::default(),
            },
        }
    }

    /// Add an artifact to the registry
    pub fn add_artifact(&mut self, artifact: ArtifactRecord) -> Result<()> {
        self.artifacts.insert(artifact.key.clone(), artifact);
        self.metadata.total_artifacts = self.artifacts.len();
        self.metadata.updated_at = SystemTime::now();
        self.update_statistics();
        Ok(())
    }

    /// Get an artifact from the registry
    pub fn get_artifact(&self, key: &str) -> Option<&ArtifactRecord> {
        self.artifacts.get(key)
    }

    /// Get a mutable reference to an artifact
    pub fn get_artifact_mut(&mut self, key: &str) -> Option<&mut ArtifactRecord> {
        self.artifacts.get_mut(key)
    }

    /// Remove an artifact from the registry
    pub fn remove_artifact(&mut self, key: &str) -> Option<ArtifactRecord> {
        let result = self.artifacts.remove(key);
        if result.is_some() {
            self.metadata.total_artifacts = self.artifacts.len();
            self.metadata.updated_at = SystemTime::now();
            self.update_statistics();
        }
        result
    }

    /// Update registry statistics
    fn update_statistics(&mut self) {
        let mut total_size = 0u64;
        let mut active_count = 0;
        let mut failed_count = 0;
        let mut deleted_count = 0;

        for artifact in self.artifacts.values() {
            total_size += artifact.metadata.size_bytes;
            match artifact.status {
                ArtifactStatus::Uploaded => active_count += 1,
                ArtifactStatus::Failed => failed_count += 1,
                ArtifactStatus::Deleted => deleted_count += 1,
                _ => {}
            }
        }

        let average_size = if !self.artifacts.is_empty() {
            total_size as f64 / self.artifacts.len() as f64
        } else {
            0.0
        };

        self.metadata.statistics = RegistryStatistics {
            total_size_bytes: total_size,
            active_artifacts: active_count,
            failed_uploads: failed_count,
            deleted_artifacts: deleted_count,
            average_size_bytes: average_size,
        };
    }
}

impl UploadManager {
    /// Create a new upload manager
    pub fn new(config: ArtifactUploadConfig) -> Self {
        Self {
            config,
            active_uploads: HashMap::new(),
            upload_history: Vec::new(),
        }
    }

    /// Create a new upload task
    pub fn create_upload_task(&mut self, local_path: &Path, remote_key: &str) -> Result<String> {
        let task_id = uuid::Uuid::new_v4().to_string();
        let file_size = fs::metadata(local_path)?.len();

        let task = UploadTask {
            id: task_id.clone(),
            local_path: local_path.to_path_buf(),
            remote_key: remote_key.to_string(),
            progress: UploadProgress {
                bytes_uploaded: 0,
                total_bytes: file_size,
                percentage: 0.0,
                speed_bps: 0.0,
                eta_seconds: None,
            },
            status: UploadStatus::Queued,
            started_at: SystemTime::now(),
            config: UploadTaskConfig {
                compress: self.config.compress,
                compression_level: self.config.compression_level,
                encrypt: self.config.encrypt,
                chunk_size_bytes: self.config.parallel_uploads.chunk_size_mb * 1024 * 1024,
                retry_attempts: 3,
            },
        };

        self.active_uploads.insert(task_id.clone(), task);
        Ok(task_id)
    }

    /// Complete an upload task
    pub fn complete_upload(&mut self, task_id: &str) -> Result<()> {
        if let Some(mut task) = self.active_uploads.remove(task_id) {
            task.status = UploadStatus::Completed;
            let duration = SystemTime::now()
                .duration_since(task.started_at)
                .unwrap_or(Duration::from_secs(0));

            let result = UploadResult {
                task_id: task_id.to_string(),
                remote_key: task.remote_key.clone(),
                status: UploadStatus::Completed,
                bytes_uploaded: task.progress.total_bytes,
                duration,
                average_speed_bps: if duration.as_secs() > 0 {
                    task.progress.total_bytes as f64 / duration.as_secs_f64()
                } else {
                    0.0
                },
                error_message: None,
                timestamp: SystemTime::now(),
            };

            self.upload_history.push(result);
        }

        Ok(())
    }
}

impl DownloadManager {
    /// Create a new download manager
    pub fn new(config: ArtifactDownloadConfig) -> Self {
        let cache_dir = config
            .cache_directory
            .clone()
            .unwrap_or_else(|| PathBuf::from("./cache"));

        Self {
            config: config.clone(),
            cache: DownloadCache {
                cache_dir,
                size_limit_bytes: config.cache_size_limit_mb.unwrap_or(1024) as u64 * 1024 * 1024,
                current_size_bytes: 0,
                entries: HashMap::new(),
                statistics: CacheStatistics::default(),
            },
            active_downloads: HashMap::new(),
        }
    }

    /// Check if artifact is in cache
    pub fn check_cache(&mut self, remote_key: &str) -> Result<Option<PathBuf>> {
        if !self.config.enable_caching {
            return Ok(None);
        }

        if let Some(entry) = self.cache.entries.get_mut(remote_key) {
            entry.last_accessed = SystemTime::now();
            entry.access_count += 1;
            self.cache.statistics.hits += 1;

            if entry.local_path.exists() {
                return Ok(Some(entry.local_path.clone()));
            } else {
                // File was deleted, remove from cache
                self.cache.entries.remove(remote_key);
            }
        }

        self.cache.statistics.misses += 1;
        Ok(None)
    }

    /// Update cache with downloaded file
    pub fn update_cache(&mut self, remote_key: &str, local_path: &Path) -> Result<()> {
        if !self.config.enable_caching {
            return Ok(());
        }

        let file_size = fs::metadata(local_path)?.len();

        // Check if we need to evict entries
        while self.cache.current_size_bytes + file_size > self.cache.size_limit_bytes {
            self.evict_cache_entry()?;
        }

        // Add to cache
        let cache_path = self.cache.cache_dir.join(remote_key);
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::copy(local_path, &cache_path)?;

        let entry = CacheEntry {
            key: remote_key.to_string(),
            local_path: cache_path,
            size_bytes: file_size,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 1,
            hit_score: 1.0,
        };

        self.cache.entries.insert(remote_key.to_string(), entry);
        self.cache.current_size_bytes += file_size;

        Ok(())
    }

    /// Remove artifact from cache
    pub fn remove_from_cache(&mut self, remote_key: &str) -> Result<()> {
        if let Some(entry) = self.cache.entries.remove(remote_key) {
            if entry.local_path.exists() {
                fs::remove_file(&entry.local_path)?;
            }
            self.cache.current_size_bytes -= entry.size_bytes;
        }
        Ok(())
    }

    /// Create a download task
    pub fn create_download_task(&mut self, remote_key: &str, local_path: &Path) -> Result<String> {
        let task_id = uuid::Uuid::new_v4().to_string();

        let task = DownloadTask {
            id: task_id.clone(),
            remote_key: remote_key.to_string(),
            local_path: local_path.to_path_buf(),
            progress: DownloadProgress {
                bytes_downloaded: 0,
                total_bytes: 0, // Will be updated when download starts
                percentage: 0.0,
                speed_bps: 0.0,
                eta_seconds: None,
            },
            status: DownloadStatus::Queued,
            started_at: SystemTime::now(),
        };

        self.active_downloads.insert(task_id.clone(), task);
        Ok(task_id)
    }

    /// Complete a download task
    pub fn complete_download(&mut self, task_id: &str) -> Result<()> {
        if let Some(mut task) = self.active_downloads.remove(task_id) {
            task.status = DownloadStatus::Completed;
        }
        Ok(())
    }

    /// Evict least recently used cache entry
    fn evict_cache_entry(&mut self) -> Result<()> {
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = SystemTime::now();

        for (key, entry) in &self.cache.entries {
            if entry.last_accessed < oldest_time {
                oldest_time = entry.last_accessed;
                oldest_key = Some(key.clone());
            }
        }

        if let Some(key) = oldest_key {
            self.remove_from_cache(&key)?;
            self.cache.statistics.evictions += 1;
        }

        Ok(())
    }
}

impl RetentionManager {
    /// Create a new retention manager
    pub fn new(policy: ArtifactRetentionPolicy) -> Self {
        Self {
            policy,
            scheduler: CleanupScheduler::new(),
            cleanup_history: Vec::new(),
        }
    }

    /// Run cleanup operation
    pub fn run_cleanup(
        &mut self,
        registry: &mut ArtifactRegistry,
        storage: &dyn ArtifactStorage,
    ) -> Result<CleanupResult> {
        let run_id = uuid::Uuid::new_v4().to_string();
        let start_time = SystemTime::now();

        let mut artifacts_deleted = 0;
        let mut artifacts_archived = 0;
        let mut artifacts_skipped = 0;
        let mut space_freed = 0u64;

        // Apply cleanup rules
        for rule in &self.scheduler.rules {
            if !rule.enabled {
                continue;
            }

            let artifacts_to_process: Vec<String> = registry
                .artifacts
                .keys()
                .filter(|key| {
                    self.should_apply_rule(
                        registry.artifacts.get(*key).expect("unwrap failed"),
                        rule,
                    )
                })
                .cloned()
                .collect();

            for artifact_key in artifacts_to_process {
                match rule.action {
                    CleanupAction::Delete => {
                        if let Some(artifact) = registry.get_artifact(&artifact_key) {
                            space_freed += artifact.metadata.size_bytes;
                            storage.delete(&artifact_key)?;
                            registry.remove_artifact(&artifact_key);
                            artifacts_deleted += 1;
                        } else {
                            artifacts_skipped += 1;
                        }
                    }
                    CleanupAction::Archive => {
                        if let Some(artifact) = registry.get_artifact_mut(&artifact_key) {
                            artifact.status = ArtifactStatus::Archived;
                            artifacts_archived += 1;
                        } else {
                            artifacts_skipped += 1;
                        }
                    }
                    CleanupAction::Log => {
                        // Log-only rule: record the match and leave the artifact
                        // untouched; it is not counted as processed.
                        log::info!(
                            "cleanup rule '{}' matched artifact '{artifact_key}' (log-only, no action taken)",
                            rule.name
                        );
                        artifacts_skipped += 1;
                    }
                    CleanupAction::MoveToStorageClass { .. }
                    | CleanupAction::Compress
                    | CleanupAction::TagForReview => {
                        // Matched a rule, but this action has no implementation
                        // yet: count it as skipped, never as processed.
                        log::warn!(
                            "cleanup rule '{}' matched artifact '{artifact_key}' but its \
                             action ({:?}) is not implemented; leaving the artifact untouched",
                            rule.name,
                            rule.action
                        );
                        artifacts_skipped += 1;
                    }
                }
            }
        }

        let artifacts_processed = artifacts_deleted + artifacts_archived;
        let duration = SystemTime::now()
            .duration_since(start_time)
            .unwrap_or(Duration::from_secs(0));

        let result = CleanupResult {
            run_id,
            timestamp: start_time,
            artifacts_processed,
            artifacts_deleted,
            artifacts_archived,
            artifacts_skipped,
            space_freed_bytes: space_freed,
            duration,
            summary: format!(
                "Processed {artifacts_processed} artifact(s) (deleted {artifacts_deleted}, \
                 archived {artifacts_archived}), skipped {artifacts_skipped}, freed \
                 {space_freed} bytes"
            ),
        };

        self.cleanup_history.push(result.clone());
        Ok(result)
    }

    /// Check if cleanup rule should be applied to artifact
    fn should_apply_rule(&self, artifact: &ArtifactRecord, rule: &CleanupRule) -> bool {
        match &rule.condition {
            CleanupCondition::Age { days } => {
                if let Ok(duration) = SystemTime::now().duration_since(artifact.uploaded_at) {
                    duration.as_secs() > (*days as u64 * 24 * 3600)
                } else {
                    false
                }
            }
            CleanupCondition::Status { status } => artifact.status == *status,
            CleanupCondition::Tag { tag } => artifact.tags.contains(tag),
            CleanupCondition::Size { max_size_gb } => {
                let size_gb = artifact.metadata.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                size_gb > *max_size_gb
            }
            // `Count` is registry-wide (depends on ranking every artifact,
            // not just this one) and `Custom` requires an expression
            // evaluator; neither is implementable from a single
            // `ArtifactRecord`. Matching `false` here is the conservative,
            // honest choice (never clean up an artifact this rule cannot
            // actually evaluate) rather than a silent no-op disguised as a
            // real check.
            CleanupCondition::Count { .. } | CleanupCondition::Custom { .. } => false,
        }
    }
}

impl Default for CleanupScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl CleanupScheduler {
    /// Create a new cleanup scheduler
    pub fn new() -> Self {
        Self {
            next_cleanup: SystemTime::now() + Duration::from_secs(24 * 3600), // Tomorrow
            interval: Duration::from_secs(24 * 3600),                         // Daily
            rules: Vec::new(),
            enabled: true,
        }
    }
}

// Default implementations

impl Default for LocalStorageConfig {
    fn default() -> Self {
        Self {
            create_dirs: true,
            file_permissions: Some(0o644),
            dir_permissions: Some(0o755),
            allow_symlinks: false,
        }
    }
}

impl Default for RegistryStatistics {
    fn default() -> Self {
        Self {
            total_size_bytes: 0,
            active_artifacts: 0,
            failed_uploads: 0,
            deleted_artifacts: 0,
            average_size_bytes: 0.0,
        }
    }
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            hit_ratio: 0.0,
            evictions: 0,
            average_file_size: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_local_storage_creation() {
        let temp_dir = TempDir::new().expect("unwrap failed");
        let storage = LocalArtifactStorage::new(temp_dir.path().to_path_buf());
        assert!(storage.validate_connection().is_ok());
    }

    #[test]
    fn test_artifact_registry() {
        let mut registry = ArtifactRegistry::new();
        assert_eq!(registry.artifacts.len(), 0);

        let artifact = ArtifactRecord {
            id: "test-id".to_string(),
            key: "test-key".to_string(),
            local_path: PathBuf::from("/tmp/test"),
            remote_key: "remote/test".to_string(),
            metadata: ArtifactMetadata {
                filename: "test.txt".to_string(),
                size_bytes: 100,
                content_type: "text/plain".to_string(),
                checksum: "abc123".to_string(),
                checksum_algorithm: ChecksumAlgorithm::SHA256,
                compression: None,
                encryption: None,
                custom_metadata: HashMap::new(),
            },
            uploaded_at: SystemTime::now(),
            last_accessed: None,
            tags: vec!["test".to_string()],
            retention_policy: "default".to_string(),
            status: ArtifactStatus::Uploaded,
        };

        registry.add_artifact(artifact).expect("unwrap failed");
        assert_eq!(registry.artifacts.len(), 1);
        assert!(registry.get_artifact("test-key").is_some());
    }

    #[test]
    fn test_upload_manager() {
        let config = ArtifactUploadConfig {
            compress: false,
            compression_level: 6,
            encrypt: false,
            timeout_sec: 300,
            max_file_size_mb: 1024,
            parallel_uploads: ParallelUploadConfig {
                enabled: false,
                max_concurrent: 1,
                chunk_size_mb: 100,
            },
        };

        let manager = UploadManager::new(config);
        assert_eq!(manager.active_uploads.len(), 0);
    }

    #[test]
    fn test_checksum_algorithms() {
        assert_ne!(ChecksumAlgorithm::SHA256, ChecksumAlgorithm::MD5);
        assert_eq!(ChecksumAlgorithm::SHA256, ChecksumAlgorithm::SHA256);
    }

    #[test]
    fn sha1_matches_known_test_vectors() {
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(
            sha1_hex(b"The quick brown fox jumps over the lazy dog"),
            "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
        );
    }

    #[test]
    fn crc32_matches_known_test_vectors() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn compute_checksum_rejects_unimplemented_blake2b() {
        let result = compute_checksum(b"data", ChecksumAlgorithm::Blake2b);
        assert!(
            result.is_err(),
            "Blake2b must fail explicitly, not fabricate a value"
        );
    }

    #[test]
    fn compute_checksum_sha512_and_sha1_and_crc32_are_real() {
        let sha512 = compute_checksum(b"abc", ChecksumAlgorithm::SHA512).expect("sha512 ok");
        assert_eq!(sha512.len(), 128); // 64 bytes hex-encoded
        let sha1 = compute_checksum(b"abc", ChecksumAlgorithm::SHA1).expect("sha1 ok");
        assert_eq!(sha1, "a9993e364706816aba3e25717850c26c9cd0d89d");
        let crc = compute_checksum(b"123456789", ChecksumAlgorithm::CRC32).expect("crc32 ok");
        assert_eq!(crc, "cbf43926");
    }

    #[test]
    fn test_artifact_status() {
        assert_eq!(ArtifactStatus::Pending, ArtifactStatus::Pending);
        assert_ne!(ArtifactStatus::Pending, ArtifactStatus::Uploaded);
    }
}
