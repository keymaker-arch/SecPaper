//! Query processing and ranking module.
//!
//! This module handles search queries, computes similarity rankings, and returns
//! top-k results. It coordinates between the embedding provider and storage layer
//! to perform semantic search.
//!
//! # Usage
//!
//! ```rust,no_run
//! use mcp_paper_search::query::{BruteForceSearchEngine, SearchQuery};
//! use mcp_paper_search::embedding::openai::OpenAIEmbedding;
//! use mcp_paper_search::storage::sqlite::SqliteStorage;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create embedding provider and storage
//! let embedding_provider = OpenAIEmbedding::new("api-key".to_string());
//! let storage = SqliteStorage::new("papers.db").await?;
//!
//! // Create search engine
//! let search_engine = BruteForceSearchEngine::new(embedding_provider, storage);
//!
//! // Execute search
//! let query = SearchQuery::new("machine learning".to_string(), Some(5), None);
//! let results = search_engine.search(&query).await?;
//!
//! // Results are sorted by descending similarity score
//! for result in results {
//!     println!("{} - Score: {:.3}", result.paper.title, result.score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Text normalization is automatically applied to queries before embedding.

#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]
use async_trait::async_trait;
use thiserror::Error;

use crate::embedding::{normalize_text, EmbeddingProvider};
use crate::models::{Paper, SearchResult};
use crate::storage::{PaperStorage, YearRange};

/// Errors that can occur during query processing.
#[derive(Debug, Error)]
pub enum QueryError {
    /// Embedding generation failed
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
    
    /// Storage access failed
    #[error("Storage error: {0}")]
    StorageError(String),
    
    /// Invalid query parameters
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    
    /// Other unexpected errors
    #[error("Unexpected query error: {0}")]
    Other(String),
}

/// Result type for query operations.
pub type QueryResult<T> = Result<T, QueryError>;

/// Search query parameters.
///
/// This struct encapsulates all the parameters for a search query,
/// including the query text, result count, and optional filters.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// The search query text (will be normalized and embedded)
    pub query: String,
    
    /// Maximum number of results to return
    pub top_k: usize,
    
    /// Optional publication year range filter
    pub year_range: Option<YearRange>,
}

impl SearchQuery {
    /// Create a new search query.
    ///
    /// # Arguments
    /// * `query` - The search query text
    /// * `top_k` - Maximum number of results to return (default: 10)
    /// * `year_range` - Optional year range filter
    pub fn new(query: String, top_k: Option<usize>, year_range: Option<YearRange>) -> Self {
        Self {
            query,
            top_k: top_k.unwrap_or(10),
            year_range,
        }
    }
}

/// Trait for search and ranking engines.
///
/// This trait defines the interface for performing semantic search over the
/// paper database. Implementations coordinate with embedding providers and
/// storage backends to compute and rank results.
#[async_trait]
pub trait SearchEngine: Send + Sync {
    /// Execute a search query and return ranked results.
    ///
    /// # Arguments
    /// * `query` - The search query parameters
    ///
    /// # Returns
    /// A vector of search results, sorted by relevance (highest score first)
    ///
    /// # Errors
    /// Returns `QueryError` if the search fails
    async fn search(&self, query: &SearchQuery) -> QueryResult<Vec<SearchResult>>;
}

/// Compute cosine similarity between two vectors.
///
/// Cosine similarity is a measure of similarity between two non-zero vectors
/// defined as the cosine of the angle between them. It ranges from -1 to 1,
/// where 1 means the vectors point in the same direction.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// The cosine similarity score
///
/// # Panics
/// Panics if the vectors have different lengths or if either vector has zero magnitude
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    assert!(norm_a > 0.0 && norm_b > 0.0, "Vector magnitude cannot be zero");
    
    dot_product / (norm_a * norm_b)
}

/// Default brute-force search engine implementation.
///
/// This implementation retrieves all papers from storage, computes cosine
/// similarity with the query embedding, and returns the top-k results.
/// It's suitable for small to medium datasets (MVP scope).
pub struct BruteForceSearchEngine<E, S>
where
    E: EmbeddingProvider,
    S: PaperStorage,
{
    /// Embedding provider for query embedding
    embedding_provider: E,
    
    /// Storage backend for paper retrieval
    storage: S,
}

impl<E, S> BruteForceSearchEngine<E, S>
where
    E: EmbeddingProvider,
    S: PaperStorage,
{
    /// Create a new brute-force search engine.
    ///
    /// # Arguments
    /// * `embedding_provider` - Provider for generating query embeddings
    /// * `storage` - Storage backend for retrieving papers
    pub fn new(embedding_provider: E, storage: S) -> Self {
        Self {
            embedding_provider,
            storage,
        }
    }
}

#[async_trait]
impl<E, S> SearchEngine for BruteForceSearchEngine<E, S>
where
    E: EmbeddingProvider,
    S: PaperStorage,
{
    async fn search(&self, query: &SearchQuery) -> QueryResult<Vec<SearchResult>> {
        // 1. Normalize and embed the query text
        let normalized_query = normalize_text(&query.query);
        let query_embedding = self.embedding_provider
            .embed(&normalized_query)
            .await
            .map_err(|e| QueryError::EmbeddingError(e.to_string()))?;
        
        // 2. Retrieve all papers from storage (with optional year filter)
        let papers = self.storage
            .get_all_papers(query.year_range)
            .await
            .map_err(|e| QueryError::StorageError(e.to_string()))?;
        
        // 3. Compute cosine similarity for each paper
        let mut results: Vec<SearchResult> = papers
            .into_iter()
            .map(|paper| {
                let paper_embedding = paper.embedding.as_ref()
                    .expect("Paper from storage must have embedding");
                let score = cosine_similarity(&query_embedding, paper_embedding);
                SearchResult::new(paper, score)
            })
            .collect();
        
        // 4. Sort by similarity score (descending - highest first)
        results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // 5. Return top-k results
        results.truncate(query.top_k);
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Author, Paper, RelevanceLevel};
    use crate::embedding::EmbeddingError;
    use crate::storage::StorageError;
    use std::sync::{Arc, Mutex};

    // Mock EmbeddingProvider for testing
    struct MockEmbeddingProvider {
        dimension: usize,
        model_name: String,
        embeddings: Arc<Mutex<Vec<Vec<f32>>>>,
        should_fail: bool,
    }

    impl MockEmbeddingProvider {
        fn new(dimension: usize) -> Self {
            Self {
                dimension,
                model_name: "mock-model".to_string(),
                embeddings: Arc::new(Mutex::new(Vec::new())),
                should_fail: false,
            }
        }

        fn with_failure() -> Self {
            Self {
                dimension: 384,
                model_name: "mock-model".to_string(),
                embeddings: Arc::new(Mutex::new(Vec::new())),
                should_fail: true,
            }
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            if self.should_fail {
                return Err(EmbeddingError::ApiError("Mock embedding failure".to_string()));
            }
            // Generate deterministic embedding based on text length
            let mut embedding = vec![0.0; self.dimension];
            let len = text.len();
            embedding[0] = (len as f32) / 100.0;
            embedding[1] = ((len % 10) as f32) / 10.0;
            self.embeddings.lock().unwrap().push(embedding.clone());
            Ok(embedding)
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            let mut results = Vec::new();
            for text in texts {
                results.push(self.embed(text).await?);
            }
            Ok(results)
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn model_name(&self) -> &str {
            &self.model_name
        }
    }

    // Mock PaperStorage for testing
    struct MockStorage {
        papers: Vec<Paper>,
        should_fail: bool,
    }

    impl MockStorage {
        fn new(papers: Vec<Paper>) -> Self {
            Self {
                papers,
                should_fail: false,
            }
        }

        fn with_failure() -> Self {
            Self {
                papers: Vec::new(),
                should_fail: true,
            }
        }
    }

    #[async_trait]
    impl PaperStorage for MockStorage {
        async fn initialize(&mut self) -> Result<(), StorageError> {
            Ok(())
        }

        async fn store_config(&mut self, _config: &crate::models::EmbeddingConfig) -> Result<(), StorageError> {
            Ok(())
        }

        async fn get_config(&self) -> Result<Option<crate::models::EmbeddingConfig>, StorageError> {
            Ok(None)
        }

        async fn insert_paper(&mut self, _paper: &Paper) -> Result<i64, StorageError> {
            Ok(1)
        }

        async fn exists_by_title(&self, _normalized_title: &str) -> Result<bool, StorageError> {
            Ok(false)
        }

        async fn get_all_papers(&self, year_range: Option<YearRange>) -> Result<Vec<Paper>, StorageError> {
            if self.should_fail {
                return Err(StorageError::QueryError("Mock storage failure".to_string()));
            }
            
            let filtered: Vec<Paper> = self.papers.iter()
                .filter(|p| {
                    if let Some(range) = year_range {
                        range.contains(p.publish_year)
                    } else {
                        true
                    }
                })
                .cloned()
                .collect();
            
            Ok(filtered)
        }

        async fn get_paper_by_id(&self, id: i64) -> Result<Paper, StorageError> {
            self.papers.iter()
                .find(|p| p.id == Some(id))
                .cloned()
                .ok_or_else(|| StorageError::NotFound(format!("Paper {} not found", id)))
        }

        async fn count_papers(&self) -> Result<usize, StorageError> {
            Ok(self.papers.len())
        }
    }

    fn create_test_paper(id: i64, title: &str, year: i32, embedding: Vec<f32>) -> Paper {
        Paper {
            id: Some(id),
            title: title.to_string(),
            authors: vec![Author {
                name: "Test Author".to_string(),
                affiliation: Some("Test University".to_string()),
            }],
            abstract_text: "Test abstract".to_string(),
            publish_year: year,
            embedding: Some(embedding),
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
        
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_query_defaults() {
        let query = SearchQuery::new("test query".to_string(), None, None);
        assert_eq!(query.top_k, 10);
        assert!(query.year_range.is_none());
    }

    #[tokio::test]
    async fn test_basic_search_with_sorting() {
        // Create test papers with different embeddings
        let papers = vec![
            create_test_paper(1, "Paper A", 2020, vec![1.0, 0.0, 0.0]),
            create_test_paper(2, "Paper B", 2021, vec![0.8, 0.6, 0.0]),  // Similar to query
            create_test_paper(3, "Paper C", 2022, vec![0.0, 1.0, 0.0]),
        ];
        
        let embedding_provider = MockEmbeddingProvider::new(3);
        let storage = MockStorage::new(papers);
        let search_engine = BruteForceSearchEngine::new(embedding_provider, storage);
        
        let query = SearchQuery::new("test query".to_string(), Some(3), None);
        let results = search_engine.search(&query).await.unwrap();
        
        // Should return all 3 papers
        assert_eq!(results.len(), 3);
        
        // Results should be sorted by score (descending)
        for i in 0..results.len() - 1 {
            assert!(results[i].score >= results[i + 1].score);
        }
        
        // Each result should have a relevance level
        for result in &results {
            assert!(result.score >= 0.0 && result.score <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_top_k_truncation() {
        // Create 5 test papers
        let papers = vec![
            create_test_paper(1, "Paper 1", 2020, vec![1.0, 0.0, 0.0]),
            create_test_paper(2, "Paper 2", 2021, vec![0.9, 0.1, 0.0]),
            create_test_paper(3, "Paper 3", 2022, vec![0.8, 0.2, 0.0]),
            create_test_paper(4, "Paper 4", 2023, vec![0.7, 0.3, 0.0]),
            create_test_paper(5, "Paper 5", 2024, vec![0.6, 0.4, 0.0]),
        ];
        
        let embedding_provider = MockEmbeddingProvider::new(3);
        let storage = MockStorage::new(papers);
        let search_engine = BruteForceSearchEngine::new(embedding_provider, storage);
        
        // Request only top 2 results
        let query = SearchQuery::new("test".to_string(), Some(2), None);
        let results = search_engine.search(&query).await.unwrap();
        
        // Should return exactly 2 papers
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_year_range_filtering() {
        // Create papers from different years
        let papers = vec![
            create_test_paper(1, "Old Paper", 2015, vec![1.0, 0.0, 0.0]),
            create_test_paper(2, "Recent Paper 1", 2020, vec![0.9, 0.1, 0.0]),
            create_test_paper(3, "Recent Paper 2", 2021, vec![0.8, 0.2, 0.0]),
            create_test_paper(4, "Newest Paper", 2025, vec![0.7, 0.3, 0.0]),
        ];
        
        let embedding_provider = MockEmbeddingProvider::new(3);
        let storage = MockStorage::new(papers);
        let search_engine = BruteForceSearchEngine::new(embedding_provider, storage);
        
        // Search with year range filter (2020-2021)
        let year_range = Some(YearRange::new(2020, 2021));
        let query = SearchQuery::new("test".to_string(), Some(10), year_range);
        let results = search_engine.search(&query).await.unwrap();
        
        // Should only return papers from 2020-2021
        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(result.paper.publish_year >= 2020 && result.paper.publish_year <= 2021);
        }
    }

    #[tokio::test]
    async fn test_empty_results() {
        // Empty storage
        let embedding_provider = MockEmbeddingProvider::new(3);
        let storage = MockStorage::new(Vec::new());
        let search_engine = BruteForceSearchEngine::new(embedding_provider, storage);
        
        let query = SearchQuery::new("test".to_string(), Some(10), None);
        let results = search_engine.search(&query).await.unwrap();
        
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_embedding_error_propagation() {
        let embedding_provider = MockEmbeddingProvider::with_failure();
        let storage = MockStorage::new(Vec::new());
        let search_engine = BruteForceSearchEngine::new(embedding_provider, storage);
        
        let query = SearchQuery::new("test".to_string(), Some(10), None);
        let result = search_engine.search(&query).await;
        
        assert!(result.is_err());
        match result.unwrap_err() {
            QueryError::EmbeddingError(_) => {},
            _ => panic!("Expected EmbeddingError"),
        }
    }

    #[tokio::test]
    async fn test_storage_error_propagation() {
        let embedding_provider = MockEmbeddingProvider::new(3);
        let storage = MockStorage::with_failure();
        let search_engine = BruteForceSearchEngine::new(embedding_provider, storage);
        
        let query = SearchQuery::new("test".to_string(), Some(10), None);
        let result = search_engine.search(&query).await;
        
        assert!(result.is_err());
        match result.unwrap_err() {
            QueryError::StorageError(_) => {},
            _ => panic!("Expected StorageError"),
        }
    }

    #[tokio::test]
    async fn test_relevance_level_assignment() {
        // Create papers that will have specific similarity scores
        let papers = vec![
            create_test_paper(1, "Identical", 2020, vec![1.0, 0.0, 0.0]),      // Score ~1.0
            create_test_paper(2, "Similar", 2021, vec![0.8, 0.6, 0.0]),         // Score ~0.8
        ];
        
        let embedding_provider = MockEmbeddingProvider::new(3);
        let storage = MockStorage::new(papers);
        let search_engine = BruteForceSearchEngine::new(embedding_provider, storage);
        
        let query = SearchQuery::new("test".to_string(), Some(10), None);
        let results = search_engine.search(&query).await.unwrap();
        
        // Verify that relevance levels are assigned based on scores
        for result in &results {
            if result.score > 0.95 {
                assert_eq!(result.relevance, RelevanceLevel::Identical);
            } else if result.score > 0.85 {
                assert_eq!(result.relevance, RelevanceLevel::HighlySimilar);
            } else if result.score > 0.70 {
                assert_eq!(result.relevance, RelevanceLevel::Similar);
            } else {
                assert_eq!(result.relevance, RelevanceLevel::Relevant);
            }
        }
    }
}
