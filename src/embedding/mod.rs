//! Embedding provider abstraction and implementations.
//!
//! This module defines the interface for text embedding generation and provides
//! implementations for various embedding services (e.g., OpenAI).
//!
//! The abstraction allows the system to swap between different embedding models
//! without changing the core logic of ingestion or search.

pub mod openai;

use async_trait::async_trait;
use thiserror::Error;

/// Errors that can occur during embedding operations.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// Network or API communication error
    #[error("API request failed: {0}")]
    ApiError(String),
    
    /// Invalid input text (e.g., empty, too long)
    #[error("Invalid input text: {0}")]
    InvalidInput(String),
    
    /// Configuration error (e.g., missing API key)
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Other unexpected errors
    #[error("Unexpected error: {0}")]
    Other(String),
}

/// Result type for embedding operations.
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

/// Trait for text embedding providers.
///
/// Implementors of this trait can generate vector embeddings from text inputs.
/// The trait is async to support API-based embedding services.
///
/// # Example Usage
/// ```ignore
/// let provider = OpenAIEmbedding::new(api_key);
/// let text = normalize_text("Research paper abstract");
/// let embedding = provider.embed(&text).await?;
/// ```
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding vector for the given text.
    ///
    /// # Arguments
    /// * `text` - The input text to embed (should be pre-normalized)
    ///
    /// # Returns
    /// A vector of f32 values representing the embedding
    ///
    /// # Errors
    /// Returns `EmbeddingError` if the embedding generation fails
    async fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>>;
    
    /// Generate embeddings for multiple texts in a single batch.
    ///
    /// This can be more efficient than calling `embed` multiple times,
    /// especially for API-based providers that support batch requests.
    ///
    /// # Arguments
    /// * `texts` - Slice of text inputs to embed
    ///
    /// # Returns
    /// A vector of embedding vectors, in the same order as the input texts
    ///
    /// # Errors
    /// Returns `EmbeddingError` if any embedding generation fails
    async fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>>;
    
    /// Get the dimension of embeddings produced by this provider.
    ///
    /// # Returns
    /// The number of dimensions in the embedding vectors
    fn dimension(&self) -> usize;
    
    /// Get the model name/identifier for this provider.
    ///
    /// # Returns
    /// A string identifying the embedding model (e.g., "text-embedding-3-small")
    fn model_name(&self) -> &str;
}

/// Normalizes text for consistent embedding generation.
///
/// This function applies the following transformations:
/// - Converts to lowercase
/// - Trims leading/trailing whitespace
/// - Collapses multiple consecutive spaces to a single space
///
/// # Arguments
/// * `text` - The raw text to normalize
///
/// # Returns
/// The normalized text string
///
/// # Example
/// ```ignore
/// let normalized = normalize_text("  Hello   World  ");
/// assert_eq!(normalized, "hello world");
/// ```
pub fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .trim()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text() {
        assert_eq!(normalize_text("Hello World"), "hello world");
        assert_eq!(normalize_text("  Multiple   Spaces  "), "multiple spaces");
        assert_eq!(normalize_text("UPPERCASE"), "uppercase");
        assert_eq!(normalize_text("   "), "");
    }
}
