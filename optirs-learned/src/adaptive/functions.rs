//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AdaptiveConfig, AdaptiveTransformerEnhancement, PositionalEncodingType};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_adaptive_transformer_creation() {
        let config = AdaptiveConfig::<f64>::default();
        let enhancement = AdaptiveTransformerEnhancement::<f64>::new(config);
        assert!(enhancement.is_ok());
    }
    #[test]
    fn test_positional_encoding_types() {
        let encoding_type = PositionalEncodingType::Learned;
        assert!(matches!(encoding_type, PositionalEncodingType::Learned));
    }
}
