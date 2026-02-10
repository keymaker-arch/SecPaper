//! Storage layer abstraction and implementations.
//!
//! This module defines the interface for persisting and retrieving paper metadata
//! and embeddings. The abstraction allows for different storage backends (SQLite,
//! vector databases, etc.) while maintaining a consistent API.

pub mod sqlite;

use async_trait::async_trait;
use thiserror::Error;

use crate::models::{Paper, EmbeddingConfig};

/// Errors that can occur during storage operations.
#[derive(Debug, Error)]
pub enum StorageError {
    /// Database connection error
    #[error("Database connection failed: {0}")]
    ConnectionError(String),
    
    /// Query execution error
    #[error("Query execution failed: {0}")]
    QueryError(String),
    
    /// Data serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Schema or migration error
    #[error("Schema error: {0}")]
    SchemaError(String),
    
    /// Record not found
    #[error("Record not found: {0}")]
    NotFound(String),
    
    /// Duplicate entry (e.g., same title already exists)
    #[error("Duplicate entry: {0}")]
    DuplicateEntry(String),
    
    /// Other unexpected errors
    #[error("Unexpected storage error: {0}")]
    Other(String),
}

/// Result type for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;

/// Year range filter for queries.
///
/// Allows filtering papers by publication year range.
#[derive(Debug, Clone, Copy)]
pub struct YearRange {
    /// Start year (inclusive)
    pub start: i32,
    
    /// End year (inclusive)
    pub end: i32,
}

impl YearRange {
    /// Create a new year range.
    ///
    /// # Arguments
    /// * `start` - Start year (inclusive)
    /// * `end` - End year (inclusive)
    pub fn new(start: i32, end: i32) -> Self {
        Self { start, end }
    }
    
    /// Check if a year falls within this range.
    pub fn contains(&self, year: i32) -> bool {
        year >= self.start && year <= self.end
    }
}

/// Trait for paper storage backends.
///
/// This trait defines the core operations needed to persist and retrieve
/// paper metadata and embeddings. Implementations can use different backends
/// (SQLite, PostgreSQL, vector databases, etc.).
#[async_trait]
pub trait PaperStorage: Send + Sync {
    /// Initialize the storage (create tables, indexes, etc.).
    ///
    /// This should be idempotent and safe to call multiple times.
    ///
    /// # Errors
    /// Returns `StorageError` if initialization fails
    async fn initialize(&mut self) -> StorageResult<()>;
    
    /// Store the embedding configuration.
    ///
    /// This configuration should be persisted to ensure consistency between
    /// ingestion and query-time embedding generation.
    ///
    /// # Arguments
    /// * `config` - The embedding configuration to store
    ///
    /// # Errors
    /// Returns `StorageError` if storage fails
    async fn store_config(&mut self, config: &EmbeddingConfig) -> StorageResult<()>;
    
    /// Retrieve the embedding configuration.
    ///
    /// # Returns
    /// The stored embedding configuration, or None if not set
    ///
    /// # Errors
    /// Returns `StorageError` if retrieval fails
    async fn get_config(&self) -> StorageResult<Option<EmbeddingConfig>>;
    
    /// Insert a new paper into storage.
    ///
    /// # Arguments
    /// * `paper` - The paper to insert (id will be assigned by the storage)
    ///
    /// # Returns
    /// The assigned paper ID
    ///
    /// # Errors
    /// Returns `StorageError::DuplicateEntry` if a paper with the same
    /// normalized title already exists, or other `StorageError` variants
    /// for other failures
    async fn insert_paper(&mut self, paper: &Paper) -> StorageResult<i64>;
    
    /// Check if a paper with the given normalized title exists.
    ///
    /// This is used for deduplication during ingestion.
    ///
    /// # Arguments
    /// * `normalized_title` - The normalized title to check
    ///
    /// # Returns
    /// `true` if a paper with this title exists, `false` otherwise
    ///
    /// # Errors
    /// Returns `StorageError` if the check fails
    async fn exists_by_title(&self, normalized_title: &str) -> StorageResult<bool>;
    
    /// Retrieve all papers with their embeddings.
    ///
    /// This is used during search to scan all papers and compute similarity scores.
    /// For MVP, we use brute-force search; future implementations may use
    /// specialized vector search methods.
    ///
    /// # Arguments
    /// * `year_range` - Optional year range filter
    ///
    /// # Returns
    /// A vector of all papers matching the filter criteria
    ///
    /// # Errors
    /// Returns `StorageError` if retrieval fails
    async fn get_all_papers(&self, year_range: Option<YearRange>) -> StorageResult<Vec<Paper>>;
    
    /// Get a paper by its ID.
    ///
    /// # Arguments
    /// * `id` - The paper ID
    ///
    /// # Returns
    /// The paper if found
    ///
    /// # Errors
    /// Returns `StorageError::NotFound` if the paper doesn't exist
    async fn get_paper_by_id(&self, id: i64) -> StorageResult<Paper>;
    
    /// Get the total count of papers in storage.
    ///
    /// # Returns
    /// The number of papers stored
    ///
    /// # Errors
    /// Returns `StorageError` if the count fails
    async fn count_papers(&self) -> StorageResult<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_year_range() {
        let range = YearRange::new(2020, 2023);
        assert!(range.contains(2020));
        assert!(range.contains(2022));
        assert!(range.contains(2023));
        assert!(!range.contains(2019));
        assert!(!range.contains(2024));
    }
}
