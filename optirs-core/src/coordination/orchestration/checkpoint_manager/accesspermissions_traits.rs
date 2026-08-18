//! # `AccessPermissions` - Trait Implementations
//!
//! This module contains trait implementations for `AccessPermissions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PermissionSet;
use super::types_15::AccessPermissions;

impl Default for AccessPermissions {
    fn default() -> Self {
        Self {
            owner: PermissionSet {
                read: true,
                write: true,
                execute: true,
                delete: true,
            },
            group: PermissionSet {
                read: true,
                write: false,
                execute: true,
                delete: false,
            },
            public: PermissionSet {
                read: false,
                write: false,
                execute: false,
                delete: false,
            },
            acl: Vec::new(),
        }
    }
}
