//! Paper provider module.
//!
//! This module defines the interface for sourcing paper metadata from various providers
//! and includes implementations for different data sources.
//!
//! The `PaperProvider` trait abstracts the source of paper data, allowing the ingestion
//! pipeline to work with different backends (JSON files, ArXiv API, Semantic Scholar, etc.)
//! without coupling to specific implementations.

use async_trait::async_trait;
use thiserror::Error;

use crate::models::Paper;

pub mod json;

/// Errors that can occur when fetching papers from a provider.
#[derive(Debug, Error)]
pub enum ProviderError {
    /// Failed to read from the data source
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Failed to parse the data format
    #[error("Parse error: {0}")]
    ParseError(String),
    
    /// API rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    
    /// Network or connection error
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Invalid configuration
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Other provider-specific errors
    #[error("Provider error: {0}")]
    Other(String),
}

/// Result type for provider operations.
pub type ProviderResult<T> = Result<T, ProviderError>;

/// Trait for sourcing paper metadata from various providers.
///
/// Implementations of this trait handle the specifics of fetching and parsing
/// paper metadata from different sources (local files, APIs, databases, etc.).
///
/// # Design Notes
///
/// - Providers should return papers without embeddings (embeddings are generated
///   by the ingestion pipeline)
/// - Providers are responsible for their own pagination, rate limiting, and error recovery
/// - Papers should have normalized metadata but don't need to be deduplicated
///   (deduplication is handled by the ingestion pipeline)
#[async_trait]
pub trait PaperProvider: Send + Sync {
    /// Fetch all available papers from this provider.
    ///
    /// This method should retrieve all papers from the data source. For large datasets,
    /// implementations may use internal batching or streaming.
    ///
    /// # Returns
    /// A vector of papers without embeddings
    ///
    /// # Errors
    /// Returns `ProviderError` if papers cannot be fetched or parsed
    async fn fetch_papers(&self) -> ProviderResult<Vec<Paper>>;
    
    /// Fetch a specific number of papers, useful for testing or incremental ingestion.
    ///
    /// # Arguments
    /// * `limit` - Maximum number of papers to fetch
    ///
    /// # Returns
    /// A vector of papers (may be fewer than `limit` if not enough are available)
    ///
    /// # Errors
    /// Returns `ProviderError` if papers cannot be fetched or parsed
    async fn fetch_papers_limit(&self, limit: usize) -> ProviderResult<Vec<Paper>> {
        let all_papers = self.fetch_papers().await?;
        Ok(all_papers.into_iter().take(limit).collect())
    }
    
    /// Get the total count of papers available from this provider.
    ///
    /// This is useful for progress tracking and resource planning.
    /// Implementations may return an estimate if exact count is expensive to compute.
    ///
    /// # Returns
    /// The total number of papers available
    ///
    /// # Errors
    /// Returns `ProviderError` if count cannot be determined
    async fn count_papers(&self) -> ProviderResult<usize> {
        // Default implementation fetches all papers and counts them
        // Providers should override this with a more efficient implementation if possible
        self.fetch_papers().await.map(|papers| papers.len())
    }
    
    /// Get a human-readable name/description of this provider.
    ///
    /// This is useful for logging and debugging.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    // Tests for provider implementations are in their respective modules
}
