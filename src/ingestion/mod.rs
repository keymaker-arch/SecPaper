//! Ingestion pipeline module.
//!
//! This module handles the offline ingestion pipeline that processes paper metadata,
//! generates embeddings, performs deduplication, and persists data to storage.
//!
//! # Usage Patterns
//!
//! ## Primary: Adding Papers to Existing Storage
//! The typical workflow is to connect to an existing database that already contains
//! papers and an embedding configuration:
//!
//! ```ignore
//! use mcp_paper_search::ingestion::IngestionPipeline;
//! use mcp_paper_search::storage::sqlite::SqliteStorage;
//! use mcp_paper_search::embedding::openai::OpenAIEmbedding;
//! use mcp_paper_search::provider::json::JsonFileProvider;
//!
//! // Connect to existing storage
//! let storage = SqliteStorage::open("papers.db").await?;
//! let embedding_provider = OpenAIEmbedding::new("your-api-key");
//! let mut pipeline = IngestionPipeline::connect(embedding_provider, storage, None).await?;
//!
//! // Add new papers from a provider
//! let provider = JsonFileProvider::from_file("new_papers.json").await?;
//! let stats = pipeline.ingest_from_provider(&provider).await?;
//! println!("Inserted: {}, Duplicates: {}", stats.inserted, stats.duplicates_skipped);
//! ```
//!
//! The pipeline automatically:
//! - Reads the embedding configuration from storage
//! - Validates that the provider matches the stored configuration
//! - Handles deduplication by checking normalized titles
//! - Generates embeddings and inserts new papers
//!
//! ## Secondary: Creating New Storage
//! When setting up a new database for the first time:
//!
//! ```ignore
//! // Initialize new storage with embedding configuration
//! let storage = SqliteStorage::open("new_database.db").await?;
//! let embedding_provider = FastEmbedProvider::new();
//! let mut pipeline = IngestionPipeline::initialize_new(embedding_provider, storage, None).await?;
//!
//! // Ingest initial batch of papers
//! let provider = JsonFileProvider::from_file("papers.json").await?;
//! pipeline.ingest_from_provider(&provider).await?;
//! ```

#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]
use thiserror::Error;

use crate::embedding::{normalize_text, EmbeddingProvider};
use crate::models::{EmbeddingConfig, Paper};
use crate::provider::{PaperProvider, ProviderError};
use crate::storage::PaperStorage;

/// Errors that can occur during ingestion.
#[derive(Debug, Error)]
pub enum IngestionError {
    /// Embedding generation failed
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
    
    /// Storage operation failed
    #[error("Storage error: {0}")]
    StorageError(String),
    
    /// Provider operation failed
    #[error("Provider error: {0}")]
    ProviderError(#[from] ProviderError),
    
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Duplicate paper detected
    #[error("Duplicate paper: {0}")]
    Duplicate(String),
    
    /// Other unexpected errors
    #[error("Ingestion error: {0}")]
    Other(String),
}

/// Result type for ingestion operations.
pub type IngestionResult<T> = Result<T, IngestionError>;

/// Statistics from an ingestion run.
///
/// This struct tracks the outcomes of processing a batch of papers.
#[derive(Debug, Default)]
pub struct IngestionStats {
    /// Total number of input papers processed
    pub total_processed: usize,
    
    /// Number of papers successfully inserted
    pub inserted: usize,
    
    /// Number of papers skipped due to deduplication
    pub duplicates_skipped: usize,
    
    /// Number of papers that failed to process
    pub failed: usize,
}

impl IngestionStats {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Record a successful insertion.
    pub fn record_inserted(&mut self) {
        self.total_processed += 1;
        self.inserted += 1;
    }
    
    /// Record a duplicate that was skipped.
    pub fn record_duplicate(&mut self) {
        self.total_processed += 1;
        self.duplicates_skipped += 1;
    }
    
    /// Record a failed processing attempt.
    pub fn record_failed(&mut self) {
        self.total_processed += 1;
        self.failed += 1;
    }
}

/// Ingestion pipeline coordinator.
///
/// This struct orchestrates the ingestion process, coordinating between
/// embedding generation and storage operations.
///
/// # Primary Use Case: Connect to Existing Storage
/// The typical workflow is to connect to an existing storage that already has
/// an embedding configuration, then add new papers to it:
///
/// ```ignore
/// let storage = SqliteStorage::open("papers.db").await?;
/// let provider = OpenAIEmbedding::new(...);
/// let mut pipeline = IngestionPipeline::connect(provider, storage).await?;
/// pipeline.ingest_from_provider(&json_provider).await?;
/// ```
///
/// # Secondary Use Case: Initialize New Storage
/// When setting up a new database, use `initialize_new()`:
///
/// ```ignore
/// let storage = SqliteStorage::open("new_papers.db").await?;
/// let provider = FastEmbedProvider::new(...);
/// let mut pipeline = IngestionPipeline::initialize_new(provider, storage).await?;
/// ```
pub struct IngestionPipeline<E, S>
where
    E: EmbeddingProvider,
    S: PaperStorage,
{
    /// Embedding provider for generating paper embeddings
    embedding_provider: E,
    
    /// Storage backend for persisting papers
    storage: S,
    
    /// Batch size for embedding generation
    batch_size: usize,
}

impl<E, S> IngestionPipeline<E, S>
where
    E: EmbeddingProvider,
    S: PaperStorage,
{
    /// Extract embedding configuration from a provider.
    fn extract_config(provider: &E) -> EmbeddingConfig {
        EmbeddingConfig {
            model_name: provider.model_name().to_string(),
            dimension: provider.dimension(),
        }
    }
    
    /// Connect to an existing storage and prepare for ingestion.
    ///
    /// **This is the primary method for normal operation.** Use this when you have
    /// an existing database with papers and want to add more papers to it.
    ///
    /// This method:
    /// 1. Reads the embedding configuration from storage
    /// 2. Validates that the provided embedding provider matches the stored config
    /// 3. Creates a pipeline ready for ingestion
    ///
    /// # Arguments
    /// * `embedding_provider` - Provider that matches the stored embedding configuration
    /// * `storage` - Storage backend containing existing papers and configuration
    /// * `batch_size` - Number of papers to process in each batch (default: 100)
    ///
    /// # Errors
    /// Returns `IngestionError::InvalidInput` if:
    /// - The storage has no embedding configuration
    /// - The provider's configuration doesn't match the stored configuration
    /// Returns other `IngestionError` variants for storage failures
    ///
    /// # Example
    /// ```ignore
    /// let storage = SqliteStorage::open("papers.db").await?;
    /// let provider = OpenAIEmbedding::new("your-api-key");
    /// let mut pipeline = IngestionPipeline::connect(provider, storage).await?;
    /// 
    /// // Now add more papers
    /// let json_provider = JsonFileProvider::from_file("new_papers.json").await?;
    /// pipeline.ingest_from_provider(&json_provider).await?;
    /// ```
    pub async fn connect(embedding_provider: E, storage: S, batch_size: Option<usize>) -> IngestionResult<Self> {
        // Read embedding config from storage
        let stored_config = storage.get_config()
            .await
            .map_err(|e| IngestionError::StorageError(e.to_string()))?;
        
        // Verify that storage has a config
        let stored_config = stored_config
            .ok_or_else(|| IngestionError::InvalidInput(
                "Storage has no embedding configuration. Use initialize_new() for new storage.".to_string()
            ))?;
        
        // Validate that provider config matches stored config
        let provider_config = Self::extract_config(&embedding_provider);
        
        if provider_config.model_name != stored_config.model_name {
            return Err(IngestionError::InvalidInput(format!(
                "Embedding model mismatch: provider uses '{}' but storage has '{}'",
                provider_config.model_name, stored_config.model_name
            )));
        }
        
        if provider_config.dimension != stored_config.dimension {
            return Err(IngestionError::InvalidInput(format!(
                "Embedding dimension mismatch: provider has {} but storage has {}",
                provider_config.dimension, stored_config.dimension
            )));
        }
        
        // Create and return pipeline
        Ok(Self {
            embedding_provider,
            storage,
            batch_size: batch_size.unwrap_or(100),
        })
    }
    
    /// Initialize new storage with the given embedding provider.
    ///
    /// **Use this only when setting up a new database.** For adding papers to
    /// an existing database, use `connect()` instead.
    ///
    /// This method:
    /// 1. Initializes the storage schema (creates tables, indexes, etc.)
    /// 2. Stores the embedding configuration from the provider
    /// 3. Creates a pipeline ready for ingestion
    ///
    /// # Arguments
    /// * `embedding_provider` - Provider for generating embeddings
    /// * `storage` - Storage backend to initialize
    /// * `batch_size` - Number of papers to process in each batch (default: 100)
    ///
    /// # Errors
    /// Returns `IngestionError` if storage initialization or config storage fails
    ///
    /// # Example
    /// ```ignore
    /// let storage = SqliteStorage::open("new_papers.db").await?;
    /// let provider = FastEmbedProvider::new();
    /// let mut pipeline = IngestionPipeline::initialize_new(provider, storage).await?;
    /// 
    /// // Now ingest papers into the new database
    /// let json_provider = JsonFileProvider::from_file("papers.json").await?;
    /// pipeline.ingest_from_provider(&json_provider).await?;
    /// ```
    pub async fn initialize_new(embedding_provider: E, mut storage: S, batch_size: Option<usize>) -> IngestionResult<Self> {
        // Initialize storage schema (create tables, indexes, etc.)
        storage.initialize()
            .await
            .map_err(|e| IngestionError::StorageError(e.to_string()))?;
        
        // Extract and store embedding configuration from provider
        let config = Self::extract_config(&embedding_provider);
        storage.store_config(&config)
            .await
            .map_err(|e| IngestionError::StorageError(e.to_string()))?;
        
        // Create and return pipeline
        Ok(Self {
            embedding_provider,
            storage,
            batch_size: batch_size.unwrap_or(100),
        })
    }
    
    /// Ingest a batch of papers.
    ///
    /// This method processes a batch of papers:
    /// 1. Normalizes paper titles for deduplication
    /// 2. Checks for duplicates
    /// 3. Generates embeddings for new papers
    /// 4. Persists papers to storage
    ///
    /// # Arguments
    /// * `papers` - Slice of papers to ingest (without embeddings)
    ///
    /// # Returns
    /// Statistics about the ingestion run
    ///
    /// # Errors
    /// Returns `IngestionError` if the batch cannot be processed
    pub async fn ingest_batch(&mut self, papers: &[Paper]) -> IngestionResult<IngestionStats> {
        let mut stats = IngestionStats::new();
        
        // Process papers in chunks according to batch_size
        for chunk in papers.chunks(self.batch_size) {
            // First pass: check for duplicates and collect papers to process
            let mut to_process: Vec<&Paper> = Vec::new();
            
            for paper in chunk {
                let normalized_title = Self::normalize_title(&paper.title);
                
                // Check if paper already exists
                let exists = self.storage.exists_by_title(&normalized_title)
                    .await
                    .map_err(|e| IngestionError::StorageError(e.to_string()))?;
                
                if exists {
                    stats.record_duplicate();
                } else {
                    to_process.push(paper);
                }
            }
            
            // Skip if no papers to process
            if to_process.is_empty() {
                continue;
            }
            
            // Generate embeddings for all papers to process
            let abstract_texts: Vec<String> = to_process.iter()
                .map(|p| normalize_text(&p.abstract_text))
                .collect();
            
            let abstract_refs: Vec<&str> = abstract_texts.iter()
                .map(|s| s.as_str())
                .collect();
            
            let embeddings = self.embedding_provider.embed_batch(&abstract_refs)
                .await
                .map_err(|e| IngestionError::EmbeddingError(e.to_string()))?;
            
            // Insert papers with embeddings
            for (paper, embedding) in to_process.iter().zip(embeddings.iter()) {
                let mut paper_with_embedding = (*paper).clone();
                paper_with_embedding.embedding = Some(embedding.clone());
                
                match self.storage.insert_paper(&paper_with_embedding).await {
                    Ok(_) => stats.record_inserted(),
                    Err(e) => {
                        eprintln!("Failed to insert paper '{}': {}", paper.title, e);
                        stats.record_failed();
                    }
                }
            }
        }
        
        Ok(stats)
    }
    
    /// Ingest a single paper.
    ///
    /// This is a convenience method for ingesting one paper at a time.
    ///
    /// # Arguments
    /// * `paper` - The paper to ingest (without embedding)
    ///
    /// # Returns
    /// The assigned paper ID if successful
    ///
    /// # Errors
    /// Returns `IngestionError::Duplicate` if the paper already exists,
    /// or other `IngestionError` variants for other failures
    pub async fn ingest_single(&mut self, paper: &Paper) -> IngestionResult<i64> {
        // Normalize title and check for duplicates
        let normalized_title = Self::normalize_title(&paper.title);
        
        let exists = self.storage.exists_by_title(&normalized_title)
            .await
            .map_err(|e| IngestionError::StorageError(e.to_string()))?;
        
        if exists {
            return Err(IngestionError::Duplicate(format!(
                "Paper with title '{}' already exists",
                paper.title
            )));
        }
        
        // Generate embedding for the abstract
        let normalized_abstract = normalize_text(&paper.abstract_text);
        let embedding = self.embedding_provider.embed(&normalized_abstract)
            .await
            .map_err(|e| IngestionError::EmbeddingError(e.to_string()))?;
        
        // Create paper with embedding
        let mut paper_with_embedding = paper.clone();
        paper_with_embedding.embedding = Some(embedding);
        
        // Insert paper and return the assigned ID
        self.storage.insert_paper(&paper_with_embedding)
            .await
            .map_err(|e| IngestionError::StorageError(e.to_string()))
    }
    
    /// Get a normalized version of a paper title for deduplication.
    ///
    /// # Arguments
    /// * `title` - The original paper title
    ///
    /// # Returns
    /// The normalized title (lowercase, trimmed, collapsed spaces)
    pub fn normalize_title(title: &str) -> String {
        normalize_text(title)
    }
    
    /// Ingest papers from a provider.
    ///
    /// This is the primary method for running the ingestion pipeline with a data source.
    /// It fetches papers from the provider and processes them in batches.
    ///
    /// Works with both usage patterns:
    /// - When connected to existing storage: Adds new papers while respecting existing embedding config
    /// - When initializing new storage: Performs initial data load
    ///
    /// # Arguments
    /// * `provider` - A reference to a paper provider that supplies paper metadata
    ///
    /// # Returns
    /// Statistics about the ingestion run
    ///
    /// # Errors
    /// Returns `IngestionError` if papers cannot be fetched from the provider
    /// or if the batch cannot be processed
    ///
    /// # Example
    /// ```ignore
    /// // Connect to existing storage
    /// let mut pipeline = IngestionPipeline::connect(embedding_provider, storage, None).await?;
    /// 
    /// // Ingest new papers
    /// let provider = JsonFilePaperProvider::from_file("new_papers.json").await?;
    /// let stats = pipeline.ingest_from_provider(&provider).await?;
    /// println!("Added {} new papers, skipped {} duplicates", 
    ///          stats.inserted, stats.duplicates_skipped);
    /// ```
    pub async fn ingest_from_provider<P>(&mut self, provider: &P) -> IngestionResult<IngestionStats>
    where
        P: PaperProvider,
    {
        // Fetch all papers from the provider
        let papers = provider.fetch_papers().await?;
        
        // Process papers using the existing batch ingestion logic
        self.ingest_batch(&papers).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Author;
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, Mutex};
    use async_trait::async_trait;

    // ===== Mock Implementations =====

    /// Mock embedding provider for testing with configurable behavior.
    #[derive(Clone)]
    struct MockEmbeddingProvider {
        model_name: String,
        dimension: usize,
        state: Arc<Mutex<MockEmbeddingState>>,
    }

    #[derive(Default)]
    struct MockEmbeddingState {
        embed_calls: Vec<String>,
        embed_batch_calls: Vec<Vec<String>>,
        should_fail: bool,
        fail_on_text: Option<String>,
    }

    impl MockEmbeddingProvider {
        fn new(model_name: &str, dimension: usize) -> Self {
            Self {
                model_name: model_name.to_string(),
                dimension,
                state: Arc::new(Mutex::new(MockEmbeddingState::default())),
            }
        }

        fn with_failure(self, should_fail: bool) -> Self {
            self.state.lock().unwrap().should_fail = should_fail;
            self
        }

        fn fail_on_text(self, text: &str) -> Self {
            self.state.lock().unwrap().fail_on_text = Some(text.to_string());
            self
        }

        fn get_embed_calls(&self) -> Vec<String> {
            self.state.lock().unwrap().embed_calls.clone()
        }

        fn get_embed_batch_calls(&self) -> Vec<Vec<String>> {
            self.state.lock().unwrap().embed_batch_calls.clone()
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, crate::embedding::EmbeddingError> {
            let mut state = self.state.lock().unwrap();
            state.embed_calls.push(text.to_string());
            
            if state.should_fail {
                return Err(crate::embedding::EmbeddingError::ApiError("Mock embed failure".to_string()));
            }
            
            if let Some(ref fail_text) = state.fail_on_text {
                if text.contains(fail_text) {
                    return Err(crate::embedding::EmbeddingError::ApiError(
                        format!("Failed on text containing '{}'", fail_text)
                    ));
                }
            }
            
            Ok(create_test_embedding(self.dimension))
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, crate::embedding::EmbeddingError> {
            let mut state = self.state.lock().unwrap();
            state.embed_batch_calls.push(texts.iter().map(|s| s.to_string()).collect());
            
            if state.should_fail {
                return Err(crate::embedding::EmbeddingError::ApiError("Mock embed_batch failure".to_string()));
            }
            
            if let Some(ref fail_text) = state.fail_on_text {
                for text in texts {
                    if text.contains(fail_text) {
                        return Err(crate::embedding::EmbeddingError::ApiError(
                            format!("Failed on text containing '{}'", fail_text)
                        ));
                    }
                }
            }
            
            Ok(texts.iter().map(|_| create_test_embedding(self.dimension)).collect())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn model_name(&self) -> &str {
            &self.model_name
        }
    }

    /// Mock storage for testing with in-memory state.
    #[derive(Clone)]
    struct MockStorage {
        state: Arc<Mutex<MockStorageState>>,
    }

    struct MockStorageState {
        papers: HashMap<i64, Paper>,
        titles: HashSet<String>,
        config: Option<EmbeddingConfig>,
        next_id: i64,
        initialized: bool,
        should_fail_insert: bool,
        fail_on_title: Option<String>,
        insert_calls: usize,
        exists_calls: usize,
    }

    impl Default for MockStorageState {
        fn default() -> Self {
            Self {
                papers: HashMap::new(),
                titles: HashSet::new(),
                config: None,
                next_id: 1,
                initialized: false,
                should_fail_insert: false,
                fail_on_title: None,
                insert_calls: 0,
                exists_calls: 0,
            }
        }
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                state: Arc::new(Mutex::new(MockStorageState::default())),
            }
        }

        fn with_config(self, config: EmbeddingConfig) -> Self {
            self.state.lock().unwrap().config = Some(config);
            self.state.lock().unwrap().initialized = true;
            self
        }

        fn with_existing_paper(self, title: &str) -> Self {
            let normalized = normalize_text(title);
            self.state.lock().unwrap().titles.insert(normalized);
            self
        }

        fn fail_insert_on_title(self, title: &str) -> Self {
            self.state.lock().unwrap().fail_on_title = Some(title.to_string());
            self
        }

        fn get_paper_count(&self) -> usize {
            self.state.lock().unwrap().papers.len()
        }

        fn get_insert_calls(&self) -> usize {
            self.state.lock().unwrap().insert_calls
        }

        fn has_title(&self, title: &str) -> bool {
            let normalized = normalize_text(title);
            self.state.lock().unwrap().titles.contains(&normalized)
        }
    }

    #[async_trait]
    impl PaperStorage for MockStorage {
        async fn initialize(&mut self) -> Result<(), crate::storage::StorageError> {
            self.state.lock().unwrap().initialized = true;
            Ok(())
        }

        async fn store_config(&mut self, config: &EmbeddingConfig) -> Result<(), crate::storage::StorageError> {
            self.state.lock().unwrap().config = Some(config.clone());
            Ok(())
        }

        async fn get_config(&self) -> Result<Option<EmbeddingConfig>, crate::storage::StorageError> {
            Ok(self.state.lock().unwrap().config.clone())
        }

        async fn insert_paper(&mut self, paper: &Paper) -> Result<i64, crate::storage::StorageError> {
            let mut state = self.state.lock().unwrap();
            state.insert_calls += 1;
            
            let normalized_title = normalize_text(&paper.title);
            
            // Check for failure injection
            if let Some(ref fail_title) = state.fail_on_title {
                if paper.title.contains(fail_title) {
                    return Err(crate::storage::StorageError::QueryError(
                        format!("Mock insert failure for title containing '{}'", fail_title)
                    ));
                }
            }
            
            // Check for duplicates
            if state.titles.contains(&normalized_title) {
                return Err(crate::storage::StorageError::DuplicateEntry(normalized_title));
            }
            
            let id = state.next_id;
            state.next_id += 1;
            
            let mut paper_with_id = paper.clone();
            paper_with_id.id = Some(id);
            
            state.papers.insert(id, paper_with_id);
            state.titles.insert(normalized_title);
            
            Ok(id)
        }

        async fn exists_by_title(&self, normalized_title: &str) -> Result<bool, crate::storage::StorageError> {
            let mut state = self.state.lock().unwrap();
            state.exists_calls += 1;
            Ok(state.titles.contains(normalized_title))
        }

        async fn get_all_papers(&self, _year_range: Option<crate::storage::YearRange>) -> Result<Vec<Paper>, crate::storage::StorageError> {
            Ok(self.state.lock().unwrap().papers.values().cloned().collect())
        }

        async fn get_paper_by_id(&self, id: i64) -> Result<Paper, crate::storage::StorageError> {
            self.state.lock().unwrap()
                .papers
                .get(&id)
                .cloned()
                .ok_or_else(|| crate::storage::StorageError::NotFound(format!("Paper {} not found", id)))
        }

        async fn count_papers(&self) -> Result<usize, crate::storage::StorageError> {
            Ok(self.state.lock().unwrap().papers.len())
        }
    }

    /// Mock paper provider for testing.
    struct MockPaperProvider {
        papers: Vec<Paper>,
        should_fail: bool,
    }

    impl MockPaperProvider {
        fn new(papers: Vec<Paper>) -> Self {
            Self {
                papers,
                should_fail: false,
            }
        }

        fn with_failure(mut self) -> Self {
            self.should_fail = true;
            self
        }
    }

    #[async_trait]
    impl PaperProvider for MockPaperProvider {
        async fn fetch_papers(&self) -> Result<Vec<Paper>, ProviderError> {
            if self.should_fail {
                return Err(ProviderError::Other("Mock provider failure".to_string()));
            }
            Ok(self.papers.clone())
        }

        async fn fetch_papers_limit(&self, limit: usize) -> Result<Vec<Paper>, ProviderError> {
            if self.should_fail {
                return Err(ProviderError::Other("Mock provider failure".to_string()));
            }
            Ok(self.papers.iter().take(limit).cloned().collect())
        }

        async fn count_papers(&self) -> Result<usize, ProviderError> {
            Ok(self.papers.len())
        }

        fn name(&self) -> &str {
            "MockProvider"
        }
    }

    // ===== Test Helper Functions =====

    fn create_test_paper(title: &str, year: i32) -> Paper {
        Paper {
            id: None,
            title: title.to_string(),
            authors: vec![
                Author {
                    name: "John Doe".to_string(),
                    affiliation: Some("Test University".to_string()),
                }
            ],
            abstract_text: format!("This is an abstract for the paper titled '{}'", title),
            publish_year: year,
            embedding: None,
        }
    }

    fn create_test_embedding(dimension: usize) -> Vec<f32> {
        (0..dimension).map(|i| (i as f32) / (dimension as f32)).collect()
    }

    fn create_test_config(model: &str, dim: usize) -> EmbeddingConfig {
        EmbeddingConfig {
            model_name: model.to_string(),
            dimension: dim,
        }
    }

    // ===== Configuration Validation Tests =====

    #[tokio::test]
    async fn test_connect_with_matching_config() {
        let config = create_test_config("test-model", 384);
        let storage = MockStorage::new().with_config(config);
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let result = IngestionPipeline::connect(provider, storage, None).await;
        assert!(result.is_ok(), "Should connect successfully with matching config");
    }

    #[tokio::test]
    async fn test_connect_with_model_name_mismatch() {
        let config = create_test_config("model-a", 384);
        let storage = MockStorage::new().with_config(config);
        let provider = MockEmbeddingProvider::new("model-b", 384);

        let result = IngestionPipeline::connect(provider, storage, None).await;
        assert!(result.is_err(), "Should fail with model name mismatch");
        
        if let Err(IngestionError::InvalidInput(msg)) = result {
            assert!(msg.contains("model-b"), "Error should mention provider model");
            assert!(msg.contains("model-a"), "Error should mention storage model");
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[tokio::test]
    async fn test_connect_with_dimension_mismatch() {
        let config = create_test_config("test-model", 384);
        let storage = MockStorage::new().with_config(config);
        let provider = MockEmbeddingProvider::new("test-model", 1536);

        let result = IngestionPipeline::connect(provider, storage, None).await;
        assert!(result.is_err(), "Should fail with dimension mismatch");
        
        if let Err(IngestionError::InvalidInput(msg)) = result {
            assert!(msg.contains("1536"), "Error should mention provider dimension");
            assert!(msg.contains("384"), "Error should mention storage dimension");
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[tokio::test]
    async fn test_connect_with_missing_config() {
        let storage = MockStorage::new(); // No config
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let result = IngestionPipeline::connect(provider, storage, None).await;
        assert!(result.is_err(), "Should fail with missing config");
        
        if let Err(IngestionError::InvalidInput(msg)) = result {
            assert!(msg.contains("no embedding configuration"), "Error should mention missing config");
            assert!(msg.contains("initialize_new"), "Error should suggest initialize_new");
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    // ===== Storage Initialization Tests =====

    #[tokio::test]
    async fn test_initialize_new_storage() {
        let storage = MockStorage::new();
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let result = IngestionPipeline::initialize_new(provider, storage.clone(), None).await;
        assert!(result.is_ok(), "Should initialize new storage successfully");

        // Verify config was stored
        let stored_config = storage.state.lock().unwrap().config.clone();
        assert!(stored_config.is_some(), "Config should be stored");
        let config = stored_config.unwrap();
        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.dimension, 384);
    }

    #[tokio::test]
    async fn test_initialize_calls_storage_initialize() {
        let storage = MockStorage::new();
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let _ = IngestionPipeline::initialize_new(provider, storage.clone(), None).await;
        
        assert!(storage.state.lock().unwrap().initialized, "Storage should be initialized");
    }

    // ===== Deduplication Tests =====

    #[tokio::test]
    async fn test_deduplication_exact_match() {
        let storage = MockStorage::new()
            .with_config(create_test_config("test-model", 384))
            .with_existing_paper("Existing Paper");
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage.clone(), None).await.unwrap();

        let papers = vec![
            create_test_paper("Existing Paper", 2024),
            create_test_paper("New Paper", 2024),
        ];

        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.total_processed, 2);
        assert_eq!(stats.duplicates_skipped, 1, "Should skip existing paper");
        assert_eq!(stats.inserted, 1, "Should insert new paper");
        assert_eq!(stats.failed, 0);
    }

    #[tokio::test]
    async fn test_deduplication_case_insensitive() {
        let storage = MockStorage::new()
            .with_config(create_test_config("test-model", 384))
            .with_existing_paper("Existing Paper");
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage.clone(), None).await.unwrap();

        let papers = vec![
            create_test_paper("EXISTING PAPER", 2024),
            create_test_paper("existing paper", 2024),
        ];

        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.duplicates_skipped, 2, "Should skip both case variations");
        assert_eq!(stats.inserted, 0);
    }

    #[tokio::test]
    async fn test_deduplication_whitespace_normalization() {
        let storage = MockStorage::new()
            .with_config(create_test_config("test-model", 384))
            .with_existing_paper("Existing Paper");
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage.clone(), None).await.unwrap();

        let papers = vec![
            create_test_paper("  Existing   Paper  ", 2024),
            create_test_paper("Existing  Paper", 2024),
        ];

        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.duplicates_skipped, 2, "Should skip whitespace variations");
        assert_eq!(stats.inserted, 0);
    }

    #[tokio::test]
    async fn test_deduplication_mixed_papers() {
        let storage = MockStorage::new()
            .with_config(create_test_config("test-model", 384))
            .with_existing_paper("Paper One")
            .with_existing_paper("Paper Two");
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage.clone(), None).await.unwrap();

        let papers = vec![
            create_test_paper("Paper One", 2024),      // Duplicate
            create_test_paper("Paper Three", 2024),    // New
            create_test_paper("PAPER TWO", 2024),      // Duplicate
            create_test_paper("Paper Four", 2024),     // New
        ];

        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.total_processed, 4);
        assert_eq!(stats.duplicates_skipped, 2);
        assert_eq!(stats.inserted, 2);
        assert_eq!(stats.failed, 0);
    }

    #[tokio::test]
    async fn test_no_embeddings_generated_for_duplicates() {
        let storage = MockStorage::new()
            .with_config(create_test_config("test-model", 384))
            .with_existing_paper("Duplicate");
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider.clone(), storage.clone(), None).await.unwrap();

        let papers = vec![
            create_test_paper("Duplicate", 2024),
            create_test_paper("New Paper", 2024),
        ];

        let _ = pipeline.ingest_batch(&papers).await.unwrap();

        // Only one batch call should happen (for the new paper)
        let batch_calls = provider.get_embed_batch_calls();
        assert_eq!(batch_calls.len(), 1, "Should have one batch embedding call");
        assert_eq!(batch_calls[0].len(), 1, "Should embed only one paper");
    }

    // ===== Batch Processing Tests =====

    #[tokio::test]
    async fn test_batch_processing_small_batch() {
        let storage = MockStorage::new().with_config(create_test_config("test-model", 384));
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider.clone(), storage.clone(), None).await.unwrap();

        let papers = vec![
            create_test_paper("Paper 1", 2024),
            create_test_paper("Paper 2", 2024),
            create_test_paper("Paper 3", 2024),
        ];

        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.inserted, 3);
        assert_eq!(storage.get_paper_count(), 3);
    }

    #[tokio::test]
    async fn test_batch_processing_respects_batch_size() {
        let storage = MockStorage::new().with_config(create_test_config("test-model", 384));
        let provider = MockEmbeddingProvider::new("test-model", 384);

        // Set batch size to 2
        let mut pipeline = IngestionPipeline::connect(provider.clone(), storage.clone(), Some(2)).await.unwrap();

        let papers = vec![
            create_test_paper("Paper 1", 2024),
            create_test_paper("Paper 2", 2024),
            create_test_paper("Paper 3", 2024),
            create_test_paper("Paper 4", 2024),
            create_test_paper("Paper 5", 2024),
        ];

        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.inserted, 5);
        
        // Should have 3 batch calls (2 + 2 + 1)
        let batch_calls = provider.get_embed_batch_calls();
        assert_eq!(batch_calls.len(), 3, "Should chunk into 3 batches");
        assert_eq!(batch_calls[0].len(), 2, "First batch should have 2 papers");
        assert_eq!(batch_calls[1].len(), 2, "Second batch should have 2 papers");
        assert_eq!(batch_calls[2].len(), 1, "Third batch should have 1 paper");
    }

    #[tokio::test]
    async fn test_batch_processing_empty_batch() {
        let storage = MockStorage::new().with_config(create_test_config("test-model", 384));
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage.clone(), None).await.unwrap();

        let papers: Vec<Paper> = vec![];
        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.inserted, 0);
        assert_eq!(stats.duplicates_skipped, 0);
        assert_eq!(stats.failed, 0);
        assert_eq!(storage.get_paper_count(), 0);
    }

    #[tokio::test]
    async fn test_embeddings_attached_to_papers() {
        let storage = MockStorage::new().with_config(create_test_config("test-model", 384));
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage.clone(), None).await.unwrap();

        let papers = vec![create_test_paper("Paper With Embedding", 2024)];
        let _ = pipeline.ingest_batch(&papers).await.unwrap();

        let stored_paper = storage.get_paper_by_id(1).await.unwrap();
        assert!(stored_paper.embedding.is_some(), "Paper should have embedding");
        assert_eq!(stored_paper.embedding.unwrap().len(), 384, "Embedding should have correct dimension");
    }

    // ===== Single Paper Ingestion Tests =====

    #[tokio::test]
    async fn test_ingest_single_success() {
        let storage = MockStorage::new().with_config(create_test_config("test-model", 384));
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider.clone(), storage.clone(), None).await.unwrap();

        let paper = create_test_paper("Single Paper", 2024);
        let result = pipeline.ingest_single(&paper).await;

        assert!(result.is_ok(), "Should insert single paper successfully");
        let paper_id = result.unwrap();
        assert_eq!(paper_id, 1, "Should return paper ID");
        
        // Verify embed() was called, not embed_batch()
        let embed_calls = provider.get_embed_calls();
        assert_eq!(embed_calls.len(), 1, "Should call embed() once");
    }

    #[tokio::test]
    async fn test_ingest_single_duplicate() {
        let storage = MockStorage::new()
            .with_config(create_test_config("test-model", 384))
            .with_existing_paper("Existing Paper");
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage.clone(), None).await.unwrap();

        let paper = create_test_paper("Existing Paper", 2024);
        let result = pipeline.ingest_single(&paper).await;

        assert!(result.is_err(), "Should fail on duplicate");
        if let Err(IngestionError::Duplicate(msg)) = result {
            assert!(msg.contains("Existing Paper"), "Error should mention paper title");
        } else {
            panic!("Expected Duplicate error");
        }
    }

    // ===== Provider Integration Tests =====

    #[tokio::test]
    async fn test_ingest_from_provider_success() {
        let storage = MockStorage::new().with_config(create_test_config("test-model", 384));
        let embedding_provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(embedding_provider, storage.clone(), None).await.unwrap();

        let papers = vec![
            create_test_paper("Paper 1", 2024),
            create_test_paper("Paper 2", 2025),
        ];
        let paper_provider = MockPaperProvider::new(papers);

        let stats = pipeline.ingest_from_provider(&paper_provider).await.unwrap();

        assert_eq!(stats.inserted, 2);
        assert_eq!(storage.get_paper_count(), 2);
    }

    #[tokio::test]
    async fn test_ingest_from_provider_error() {
        let storage = MockStorage::new().with_config(create_test_config("test-model", 384));
        let embedding_provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(embedding_provider, storage.clone(), None).await.unwrap();

        let paper_provider = MockPaperProvider::new(vec![]).with_failure();

        let result = pipeline.ingest_from_provider(&paper_provider).await;

        assert!(result.is_err(), "Should propagate provider error");
        if let Err(IngestionError::ProviderError(_)) = result {
            // Expected
        } else {
            panic!("Expected ProviderError");
        }
    }

    // ===== Error Handling Tests =====

    #[tokio::test]
    async fn test_embedding_error_increments_failed() {
        let storage = MockStorage::new().with_config(create_test_config("test-model", 384));
        let provider = MockEmbeddingProvider::new("test-model", 384).with_failure(true);

        let mut pipeline = IngestionPipeline::connect(provider, storage.clone(), None).await.unwrap();

        let papers = vec![create_test_paper("Paper", 2024)];
        let result = pipeline.ingest_batch(&papers).await;

        // Embedding failure should cause the batch to fail
        assert!(result.is_err(), "Should fail when embedding generation fails");
    }

    #[tokio::test]
    async fn test_partial_batch_failure() {
        let storage = MockStorage::new()
            .with_config(create_test_config("test-model", 384))
            .fail_insert_on_title("Fail");
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage.clone(), None).await.unwrap();

        let papers = vec![
            create_test_paper("Success 1", 2024),
            create_test_paper("Fail Paper", 2024),
            create_test_paper("Success 2", 2024),
        ];

        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.total_processed, 3);
        assert_eq!(stats.inserted, 2, "Should insert successful papers");
        assert_eq!(stats.failed, 1, "Should count failed paper");
        assert_eq!(storage.get_paper_count(), 2, "Only successful papers stored");
    }

    #[tokio::test]
    async fn test_storage_error_increments_failed() {
        let storage = MockStorage::new()
            .with_config(create_test_config("test-model", 384))
            .fail_insert_on_title("Bad");
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage.clone(), None).await.unwrap();

        let papers = vec![create_test_paper("Bad Paper", 2024)];
        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.failed, 1);
        assert_eq!(stats.inserted, 0);
    }

    // ===== Statistics Tests =====

    #[tokio::test]
    async fn test_stats_all_inserted() {
        let storage = MockStorage::new().with_config(create_test_config("test-model", 384));
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage, None).await.unwrap();

        let papers = vec![
            create_test_paper("Paper 1", 2024),
            create_test_paper("Paper 2", 2024),
            create_test_paper("Paper 3", 2024),
        ];

        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.total_processed, 3);
        assert_eq!(stats.inserted, 3);
        assert_eq!(stats.duplicates_skipped, 0);
        assert_eq!(stats.failed, 0);
    }

    #[tokio::test]
    async fn test_stats_all_duplicates() {
        let storage = MockStorage::new()
            .with_config(create_test_config("test-model", 384))
            .with_existing_paper("Paper 1")
            .with_existing_paper("Paper 2");
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage, None).await.unwrap();

        let papers = vec![
            create_test_paper("Paper 1", 2024),
            create_test_paper("Paper 2", 2024),
        ];

        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.total_processed, 2);
        assert_eq!(stats.inserted, 0);
        assert_eq!(stats.duplicates_skipped, 2);
        assert_eq!(stats.failed, 0);
    }

    #[tokio::test]
    async fn test_stats_mixed_outcomes() {
        let storage = MockStorage::new()
            .with_config(create_test_config("test-model", 384))
            .with_existing_paper("Duplicate")
            .fail_insert_on_title("Fail");
        let provider = MockEmbeddingProvider::new("test-model", 384);

        let mut pipeline = IngestionPipeline::connect(provider, storage, None).await.unwrap();

        let papers = vec![
            create_test_paper("Success", 2024),      // Inserted
            create_test_paper("Duplicate", 2024),    // Skipped
            create_test_paper("Fail Paper", 2024),   // Failed
        ];

        let stats = pipeline.ingest_batch(&papers).await.unwrap();

        assert_eq!(stats.total_processed, 3);
        assert_eq!(stats.inserted, 1);
        assert_eq!(stats.duplicates_skipped, 1);
        assert_eq!(stats.failed, 1);
    }

    #[tokio::test]
    async fn test_stats_record_methods() {
        let mut stats = IngestionStats::new();

        stats.record_inserted();
        stats.record_duplicate();
        stats.record_failed();

        assert_eq!(stats.total_processed, 3);
        assert_eq!(stats.inserted, 1);
        assert_eq!(stats.duplicates_skipped, 1);
        assert_eq!(stats.failed, 1);
    }

    // ===== Text Normalization Tests =====

    #[tokio::test]
    async fn test_normalize_title_lowercase() {
        let normalized = IngestionPipeline::<MockEmbeddingProvider, MockStorage>::normalize_title("UPPERCASE TITLE");
        assert_eq!(normalized, "uppercase title");
    }

    #[tokio::test]
    async fn test_normalize_title_whitespace() {
        let normalized = IngestionPipeline::<MockEmbeddingProvider, MockStorage>::normalize_title("  Multiple   Spaces  ");
        assert_eq!(normalized, "multiple spaces");
    }

    #[tokio::test]
    async fn test_normalize_title_empty() {
        let normalized = IngestionPipeline::<MockEmbeddingProvider, MockStorage>::normalize_title("   ");
        assert_eq!(normalized, "");
    }

    #[tokio::test]
    async fn test_normalize_title_consistency() {
        let title1 = IngestionPipeline::<MockEmbeddingProvider, MockStorage>::normalize_title("Test Paper");
        let title2 = normalize_text("Test Paper");
        assert_eq!(title1, title2, "normalize_title should match normalize_text");
    }
}
