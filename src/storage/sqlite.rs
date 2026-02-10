//! SQLite storage implementation.
//!
//! This module provides a SQLite-based implementation of the `PaperStorage` trait.
//! It uses rusqlite for database access and stores embeddings as BLOBs.

use super::{PaperStorage, StorageError, StorageResult, YearRange};
use crate::embedding::normalize_text;
use crate::models::{Author, EmbeddingConfig, Paper};
use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension};
use std::sync::{Arc, Mutex};

/// SQLite-based paper storage.
///
/// This implementation stores papers and their embeddings in a SQLite database.
/// Embeddings are serialized as float32 arrays and stored in BLOB columns.
///
/// # Schema
/// The database contains two main tables:
/// - `config`: Stores embedding configuration (model name, dimension)
/// - `papers`: Stores paper metadata and embeddings
pub struct SqliteStorage {
    /// Path to the SQLite database file
    db_path: String,
    
    /// Database connection (wrapped in Arc<Mutex> for interior mutability)
    connection: Option<Arc<Mutex<Connection>>>,
}

impl SqliteStorage {
    /// Create a new SQLite storage instance.
    ///
    /// # Arguments
    /// * `db_path` - Path to the SQLite database file
    pub fn new(db_path: String) -> Self {
        Self {
            db_path,
            connection: None,
        }
    }
    
    /// Create the database schema.
    ///
    /// This creates the following tables:
    /// - `config`: (key TEXT PRIMARY KEY, value TEXT)
    /// - `papers`: (id INTEGER PRIMARY KEY, title TEXT, authors TEXT,
    ///              abstract TEXT, publish_year INTEGER, embedding BLOB)
    ///
    /// # Errors
    /// Returns `StorageError` if schema creation fails
    fn create_schema(&self) -> StorageResult<()> {
        let conn = self.connection.as_ref()
            .ok_or_else(|| StorageError::ConnectionError("Not connected".to_string()))?;
        
        let conn = conn.lock()
            .map_err(|e| StorageError::ConnectionError(format!("Lock error: {}", e)))?;
        
        // Create config table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
            [],
        ).map_err(|e| StorageError::SchemaError(format!("Failed to create config table: {}", e)))?;
        
        // Create papers table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                abstract TEXT NOT NULL,
                publish_year INTEGER NOT NULL,
                embedding BLOB NOT NULL
            )",
            [],
        ).map_err(|e| StorageError::SchemaError(format!("Failed to create papers table: {}", e)))?;
        
        // Create index on title for faster deduplication checks
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)",
            [],
        ).map_err(|e| StorageError::SchemaError(format!("Failed to create index: {}", e)))?;
        
        Ok(())
    }
    
    /// Serialize an embedding vector to bytes for BLOB storage.
    ///
    /// # Arguments
    /// * `embedding` - The embedding vector to serialize
    ///
    /// # Returns
    /// A byte vector containing the serialized embedding
    fn serialize_embedding(embedding: &[f32]) -> Vec<u8> {
        // Convert f32 slice to bytes (little-endian)
        let mut bytes = Vec::with_capacity(embedding.len() * 4);
        for &value in embedding {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
    
    /// Deserialize an embedding vector from BLOB bytes.
    ///
    /// # Arguments
    /// * `bytes` - The byte slice containing the serialized embedding
    ///
    /// # Returns
    /// The deserialized embedding vector
    ///
    /// # Errors
    /// Returns error if deserialization fails or if the byte length is invalid
    fn deserialize_embedding(bytes: &[u8]) -> StorageResult<Vec<f32>> {
        // Check if byte length is valid (must be multiple of 4 for f32)
        if bytes.len() % 4 != 0 {
            return Err(StorageError::SerializationError(
                format!("Invalid byte length: {} is not a multiple of 4", bytes.len())
            ));
        }
        
        let mut embedding = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3]
            ]);
            embedding.push(value);
        }
        Ok(embedding)
    }
}

#[async_trait]
impl PaperStorage for SqliteStorage {
    async fn initialize(&mut self) -> StorageResult<()> {
        // Open database connection
        let conn = Connection::open(&self.db_path)
            .map_err(|e| StorageError::ConnectionError(format!("Failed to open database: {}", e)))?;
        
        self.connection = Some(Arc::new(Mutex::new(conn)));
        
        // Create schema
        self.create_schema()?;
        
        Ok(())
    }
    
    async fn store_config(&mut self, config: &EmbeddingConfig) -> StorageResult<()> {
        let conn = self.connection.as_ref()
            .ok_or_else(|| StorageError::ConnectionError("Not connected".to_string()))?;
        
        let conn = conn.lock()
            .map_err(|e| StorageError::ConnectionError(format!("Lock error: {}", e)))?;
        
        // Serialize config to JSON
        let config_json = serde_json::to_string(config)
            .map_err(|e| StorageError::SerializationError(format!("Failed to serialize config: {}", e)))?;
        
        // Insert or replace config
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?1, ?2)",
            params!["embedding_config", config_json],
        ).map_err(|e| StorageError::QueryError(format!("Failed to store config: {}", e)))?;
        
        Ok(())
    }
    
    async fn get_config(&self) -> StorageResult<Option<EmbeddingConfig>> {
        let conn = self.connection.as_ref()
            .ok_or_else(|| StorageError::ConnectionError("Not connected".to_string()))?;
        
        let conn = conn.lock()
            .map_err(|e| StorageError::ConnectionError(format!("Lock error: {}", e)))?;
        
        // Query config from database
        let config_json: Option<String> = conn.query_row(
            "SELECT value FROM config WHERE key = ?1",
            params!["embedding_config"],
            |row| row.get(0),
        ).optional()
        .map_err(|e| StorageError::QueryError(format!("Failed to retrieve config: {}", e)))?;
        
        // Deserialize if found
        if let Some(json) = config_json {
            let config: EmbeddingConfig = serde_json::from_str(&json)
                .map_err(|e| StorageError::SerializationError(format!("Failed to deserialize config: {}", e)))?;
            Ok(Some(config))
        } else {
            Ok(None)
        }
    }
    
    async fn insert_paper(&mut self, paper: &Paper) -> StorageResult<i64> {
        let conn = self.connection.as_ref()
            .ok_or_else(|| StorageError::ConnectionError("Not connected".to_string()))?;
        
        let conn = conn.lock()
            .map_err(|e| StorageError::ConnectionError(format!("Lock error: {}", e)))?;
        
        // Normalize title for deduplication
        let normalized_title = normalize_text(&paper.title);
        
        // Check for duplicate
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM papers WHERE title = ?1)",
            params![&normalized_title],
            |row| row.get(0),
        ).map_err(|e| StorageError::QueryError(format!("Failed to check for duplicate: {}", e)))?;
        
        if exists {
            return Err(StorageError::DuplicateEntry(format!("Paper with title '{}' already exists", paper.title)));
        }
        
        // Serialize authors to JSON
        let authors_json = serde_json::to_string(&paper.authors)
            .map_err(|e| StorageError::SerializationError(format!("Failed to serialize authors: {}", e)))?;
        
        // Serialize embedding
        let embedding_blob = if let Some(ref emb) = paper.embedding {
            Self::serialize_embedding(emb)
        } else {
            return Err(StorageError::SerializationError("Paper must have an embedding".to_string()));
        };
        
        // Insert paper
        conn.execute(
            "INSERT INTO papers (title, authors, abstract, publish_year, embedding) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![&normalized_title, &authors_json, &paper.abstract_text, &paper.publish_year, &embedding_blob],
        ).map_err(|e| StorageError::QueryError(format!("Failed to insert paper: {}", e)))?;
        
        Ok(conn.last_insert_rowid())
    }
    
    async fn exists_by_title(&self, normalized_title: &str) -> StorageResult<bool> {
        let conn = self.connection.as_ref()
            .ok_or_else(|| StorageError::ConnectionError("Not connected".to_string()))?;
        
        let conn = conn.lock()
            .map_err(|e| StorageError::ConnectionError(format!("Lock error: {}", e)))?;
        
        // Normalize the input title for consistent comparison
        let normalized = normalize_text(normalized_title);
        
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM papers WHERE title = ?1)",
            params![&normalized],
            |row| row.get(0),
        ).map_err(|e| StorageError::QueryError(format!("Failed to check title existence: {}", e)))?;
        
        Ok(exists)
    }
    
    async fn get_all_papers(&self, year_range: Option<YearRange>) -> StorageResult<Vec<Paper>> {
        let conn = self.connection.as_ref()
            .ok_or_else(|| StorageError::ConnectionError("Not connected".to_string()))?;
        
        let conn = conn.lock()
            .map_err(|e| StorageError::ConnectionError(format!("Lock error: {}", e)))?;
        
        // Build query based on year filter
        let (query, params): (String, Vec<Box<dyn rusqlite::ToSql>>) = if let Some(range) = year_range {
            (
                "SELECT id, title, authors, abstract, publish_year, embedding FROM papers WHERE publish_year >= ?1 AND publish_year <= ?2".to_string(),
                vec![Box::new(range.start), Box::new(range.end)],
            )
        } else {
            (
                "SELECT id, title, authors, abstract, publish_year, embedding FROM papers".to_string(),
                vec![],
            )
        };
        
        let mut stmt = conn.prepare(&query)
            .map_err(|e| StorageError::QueryError(format!("Failed to prepare query: {}", e)))?;
        
        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        
        let papers = stmt.query_map(param_refs.as_slice(), |row| {
            let id: i64 = row.get(0)?;
            let title: String = row.get(1)?;
            let authors_json: String = row.get(2)?;
            let abstract_text: String = row.get(3)?;
            let publish_year: i32 = row.get(4)?;
            let embedding_blob: Vec<u8> = row.get(5)?;
            
            Ok((id, title, authors_json, abstract_text, publish_year, embedding_blob))
        }).map_err(|e| StorageError::QueryError(format!("Failed to query papers: {}", e)))?;
        
        let mut result = Vec::new();
        for paper_row in papers {
            let (id, title, authors_json, abstract_text, publish_year, embedding_blob) = paper_row
                .map_err(|e| StorageError::QueryError(format!("Failed to read paper row: {}", e)))?;
            
            // Deserialize authors
            let authors: Vec<Author> = serde_json::from_str(&authors_json)
                .map_err(|e| StorageError::SerializationError(format!("Failed to deserialize authors: {}", e)))?;
            
            // Deserialize embedding
            let embedding = Self::deserialize_embedding(&embedding_blob)?;
            
            result.push(Paper {
                id: Some(id),
                title,
                authors,
                abstract_text,
                publish_year,
                embedding: Some(embedding),
            });
        }
        
        Ok(result)
    }
    
    async fn get_paper_by_id(&self, id: i64) -> StorageResult<Paper> {
        let conn = self.connection.as_ref()
            .ok_or_else(|| StorageError::ConnectionError("Not connected".to_string()))?;
        
        let conn = conn.lock()
            .map_err(|e| StorageError::ConnectionError(format!("Lock error: {}", e)))?;
        
        let result = conn.query_row(
            "SELECT id, title, authors, abstract, publish_year, embedding FROM papers WHERE id = ?1",
            params![id],
            |row| {
                let id: i64 = row.get(0)?;
                let title: String = row.get(1)?;
                let authors_json: String = row.get(2)?;
                let abstract_text: String = row.get(3)?;
                let publish_year: i32 = row.get(4)?;
                let embedding_blob: Vec<u8> = row.get(5)?;
                
                Ok((id, title, authors_json, abstract_text, publish_year, embedding_blob))
            },
        ).optional()
        .map_err(|e| StorageError::QueryError(format!("Failed to query paper by ID: {}", e)))?;
        
        if let Some((id, title, authors_json, abstract_text, publish_year, embedding_blob)) = result {
            // Deserialize authors
            let authors: Vec<Author> = serde_json::from_str(&authors_json)
                .map_err(|e| StorageError::SerializationError(format!("Failed to deserialize authors: {}", e)))?;
            
            // Deserialize embedding
            let embedding = Self::deserialize_embedding(&embedding_blob)?;
            
            Ok(Paper {
                id: Some(id),
                title,
                authors,
                abstract_text,
                publish_year,
                embedding: Some(embedding),
            })
        } else {
            Err(StorageError::NotFound(format!("Paper with ID {} not found", id)))
        }
    }
    
    async fn count_papers(&self) -> StorageResult<usize> {
        let conn = self.connection.as_ref()
            .ok_or_else(|| StorageError::ConnectionError("Not connected".to_string()))?;
        
        let conn = conn.lock()
            .map_err(|e| StorageError::ConnectionError(format!("Lock error: {}", e)))?;
        
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM papers",
            [],
            |row| row.get(0),
        ).map_err(|e| StorageError::QueryError(format!("Failed to count papers: {}", e)))?;
        
        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Author, EmbeddingConfig};
    use tempfile::NamedTempFile;
    
    /// Create a test storage instance with a temporary database.
    async fn create_test_storage() -> (SqliteStorage, NamedTempFile) {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let db_path = temp_file.path().to_str().unwrap().to_string();
        let mut storage = SqliteStorage::new(db_path);
        storage.initialize().await.expect("Failed to initialize storage");
        (storage, temp_file)
    }
    
    /// Create a test paper with a given title and year.
    fn create_test_paper(title: &str, year: i32, embedding: Vec<f32>) -> Paper {
        Paper {
            id: None,
            title: title.to_string(),
            authors: vec![
                Author {
                    name: "John Doe".to_string(),
                    affiliation: Some("MIT".to_string()),
                },
                Author {
                    name: "Jane Smith".to_string(),
                    affiliation: Some("Stanford".to_string()),
                },
            ],
            abstract_text: "This is a test abstract about testing.".to_string(),
            publish_year: year,
            embedding: Some(embedding),
        }
    }
    
    /// Create a test embedding vector with given dimension.
    fn create_test_embedding(dimension: usize) -> Vec<f32> {
        (0..dimension).map(|i| (i as f32) * 0.1).collect()
    }
    
    #[tokio::test]
    async fn test_initialization() {
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap().to_string();
        let mut storage = SqliteStorage::new(db_path);
        
        // Should succeed
        let result = storage.initialize().await;
        assert!(result.is_ok());
        
        // Should be idempotent (safe to call multiple times)
        let result = storage.initialize().await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_store_and_retrieve_config() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let config = EmbeddingConfig {
            model_name: "text-embedding-3-small".to_string(),
            dimension: 1536,
        };
        
        // Store config
        storage.store_config(&config).await.expect("Failed to store config");
        
        // Retrieve config
        let retrieved = storage.get_config().await.expect("Failed to get config");
        assert!(retrieved.is_some());
        
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.model_name, "text-embedding-3-small");
        assert_eq!(retrieved.dimension, 1536);
    }
    
    #[tokio::test]
    async fn test_config_not_found() {
        let (storage, _temp_file) = create_test_storage().await;
        
        // Config should be None if not set
        let config = storage.get_config().await.expect("Failed to get config");
        assert!(config.is_none());
    }
    
    #[tokio::test]
    async fn test_config_update() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        // Store initial config
        let config1 = EmbeddingConfig {
            model_name: "model-v1".to_string(),
            dimension: 512,
        };
        storage.store_config(&config1).await.unwrap();
        
        // Update config
        let config2 = EmbeddingConfig {
            model_name: "model-v2".to_string(),
            dimension: 1024,
        };
        storage.store_config(&config2).await.unwrap();
        
        // Should retrieve the updated config
        let retrieved = storage.get_config().await.unwrap().unwrap();
        assert_eq!(retrieved.model_name, "model-v2");
        assert_eq!(retrieved.dimension, 1024);
    }
    
    #[tokio::test]
    async fn test_insert_paper() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let embedding = create_test_embedding(128);
        let paper = create_test_paper("Test Paper", 2023, embedding);
        
        // Insert paper
        let id = storage.insert_paper(&paper).await.expect("Failed to insert paper");
        assert!(id > 0);
        
        // Verify it was inserted
        let count = storage.count_papers().await.unwrap();
        assert_eq!(count, 1);
    }
    
    #[tokio::test]
    async fn test_insert_paper_without_embedding() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let mut paper = create_test_paper("Test Paper", 2023, vec![]);
        paper.embedding = None; // No embedding
        
        // Should fail without embedding
        let result = storage.insert_paper(&paper).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::SerializationError(_)));
    }
    
    #[tokio::test]
    async fn test_duplicate_detection() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let embedding = create_test_embedding(128);
        let paper1 = create_test_paper("Duplicate Title", 2023, embedding.clone());
        
        // Insert first paper
        storage.insert_paper(&paper1).await.expect("Failed to insert first paper");
        
        // Try to insert duplicate (same title, different year)
        let paper2 = create_test_paper("Duplicate Title", 2024, embedding.clone());
        let result = storage.insert_paper(&paper2).await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::DuplicateEntry(_)));
    }
    
    #[tokio::test]
    async fn test_duplicate_detection_case_insensitive() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let embedding = create_test_embedding(128);
        
        // Insert with lowercase title
        let paper1 = create_test_paper("test paper", 2023, embedding.clone());
        storage.insert_paper(&paper1).await.unwrap();
        
        // Try to insert with uppercase (should be considered duplicate)
        let paper2 = create_test_paper("TEST PAPER", 2024, embedding.clone());
        let result = storage.insert_paper(&paper2).await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::DuplicateEntry(_)));
    }
    
    #[tokio::test]
    async fn test_duplicate_detection_whitespace_normalized() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let embedding = create_test_embedding(128);
        
        // Insert with extra whitespace
        let paper1 = create_test_paper("  Test   Paper  ", 2023, embedding.clone());
        storage.insert_paper(&paper1).await.unwrap();
        
        // Try to insert with different whitespace (should be considered duplicate)
        let paper2 = create_test_paper("Test Paper", 2024, embedding.clone());
        let result = storage.insert_paper(&paper2).await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::DuplicateEntry(_)));
    }
    
    #[tokio::test]
    async fn test_exists_by_title() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let embedding = create_test_embedding(128);
        let paper = create_test_paper("Existing Paper", 2023, embedding);
        
        // Initially should not exist
        let exists = storage.exists_by_title("existing paper").await.unwrap();
        assert!(!exists);
        
        // Insert paper
        storage.insert_paper(&paper).await.unwrap();
        
        // Now should exist (case-insensitive and normalized)
        let exists = storage.exists_by_title("existing paper").await.unwrap();
        assert!(exists);
        
        let exists = storage.exists_by_title("EXISTING PAPER").await.unwrap();
        assert!(exists);
        
        let exists = storage.exists_by_title("  existing   paper  ").await.unwrap();
        assert!(exists);
        
        // Different title should not exist
        let exists = storage.exists_by_title("different paper").await.unwrap();
        assert!(!exists);
    }
    
    #[tokio::test]
    async fn test_get_paper_by_id() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let embedding = create_test_embedding(128);
        let paper = create_test_paper("Test Paper", 2023, embedding.clone());
        
        // Insert paper
        let id = storage.insert_paper(&paper).await.unwrap();
        
        // Retrieve by ID
        let retrieved = storage.get_paper_by_id(id).await.expect("Failed to retrieve paper");
        
        assert_eq!(retrieved.id, Some(id));
        assert_eq!(retrieved.title, "test paper"); // Normalized
        assert_eq!(retrieved.authors.len(), 2);
        assert_eq!(retrieved.authors[0].name, "John Doe");
        assert_eq!(retrieved.authors[1].name, "Jane Smith");
        assert_eq!(retrieved.abstract_text, "This is a test abstract about testing.");
        assert_eq!(retrieved.publish_year, 2023);
        assert_eq!(retrieved.embedding.as_ref().unwrap(), &embedding);
    }
    
    #[tokio::test]
    async fn test_get_paper_by_id_not_found() {
        let (storage, _temp_file) = create_test_storage().await;
        
        // Try to retrieve non-existent paper
        let result = storage.get_paper_by_id(999).await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::NotFound(_)));
    }
    
    #[tokio::test]
    async fn test_get_all_papers() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        // Insert multiple papers
        let embedding = create_test_embedding(128);
        
        let paper1 = create_test_paper("Paper 1", 2021, embedding.clone());
        let paper2 = create_test_paper("Paper 2", 2022, embedding.clone());
        let paper3 = create_test_paper("Paper 3", 2023, embedding.clone());
        
        storage.insert_paper(&paper1).await.unwrap();
        storage.insert_paper(&paper2).await.unwrap();
        storage.insert_paper(&paper3).await.unwrap();
        
        // Get all papers
        let papers = storage.get_all_papers(None).await.unwrap();
        
        assert_eq!(papers.len(), 3);
        assert!(papers.iter().all(|p| p.embedding.is_some()));
    }
    
    #[tokio::test]
    async fn test_get_all_papers_empty() {
        let (storage, _temp_file) = create_test_storage().await;
        
        // Get all papers from empty database
        let papers = storage.get_all_papers(None).await.unwrap();
        assert_eq!(papers.len(), 0);
    }
    
    #[tokio::test]
    async fn test_get_all_papers_with_year_filter() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let embedding = create_test_embedding(128);
        
        // Insert papers from different years
        let paper1 = create_test_paper("Paper 2020", 2020, embedding.clone());
        let paper2 = create_test_paper("Paper 2021", 2021, embedding.clone());
        let paper3 = create_test_paper("Paper 2022", 2022, embedding.clone());
        let paper4 = create_test_paper("Paper 2023", 2023, embedding.clone());
        
        storage.insert_paper(&paper1).await.unwrap();
        storage.insert_paper(&paper2).await.unwrap();
        storage.insert_paper(&paper3).await.unwrap();
        storage.insert_paper(&paper4).await.unwrap();
        
        // Filter for papers from 2021-2022
        let year_range = Some(YearRange::new(2021, 2022));
        let papers = storage.get_all_papers(year_range).await.unwrap();
        
        assert_eq!(papers.len(), 2);
        assert!(papers.iter().all(|p| p.publish_year >= 2021 && p.publish_year <= 2022));
    }
    
    #[tokio::test]
    async fn test_year_filter_boundaries() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let embedding = create_test_embedding(128);
        
        // Insert papers at boundary years
        let paper1 = create_test_paper("Paper 2019", 2019, embedding.clone());
        let paper2 = create_test_paper("Paper 2020", 2020, embedding.clone());
        let paper3 = create_test_paper("Paper 2023", 2023, embedding.clone());
        let paper4 = create_test_paper("Paper 2024", 2024, embedding.clone());
        
        storage.insert_paper(&paper1).await.unwrap();
        storage.insert_paper(&paper2).await.unwrap();
        storage.insert_paper(&paper3).await.unwrap();
        storage.insert_paper(&paper4).await.unwrap();
        
        // Filter for 2020-2023 (inclusive boundaries)
        let year_range = Some(YearRange::new(2020, 2023));
        let papers = storage.get_all_papers(year_range).await.unwrap();
        
        assert_eq!(papers.len(), 2);
        let years: Vec<i32> = papers.iter().map(|p| p.publish_year).collect();
        assert!(years.contains(&2020));
        assert!(years.contains(&2023));
        assert!(!years.contains(&2019));
        assert!(!years.contains(&2024));
    }
    
    #[tokio::test]
    async fn test_count_papers() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        // Initially empty
        let count = storage.count_papers().await.unwrap();
        assert_eq!(count, 0);
        
        // Insert papers
        let embedding = create_test_embedding(128);
        
        for i in 0..5 {
            let paper = create_test_paper(&format!("Paper {}", i), 2023, embedding.clone());
            storage.insert_paper(&paper).await.unwrap();
        }
        
        // Count should be 5
        let count = storage.count_papers().await.unwrap();
        assert_eq!(count, 5);
    }
    
    #[tokio::test]
    async fn test_embedding_serialization_deserialization() {
        // Test with various embedding sizes
        for dimension in [128, 512, 1536, 3072] {
            let embedding: Vec<f32> = (0..dimension).map(|i| i as f32 * 0.001).collect();
            
            let serialized = SqliteStorage::serialize_embedding(&embedding);
            let deserialized = SqliteStorage::deserialize_embedding(&serialized).unwrap();
            
            assert_eq!(embedding.len(), deserialized.len());
            for (original, restored) in embedding.iter().zip(deserialized.iter()) {
                assert_eq!(original, restored);
            }
        }
    }
    
    #[tokio::test]
    async fn test_embedding_with_special_values() {
        // Test with special float values
        let embedding = vec![0.0, -0.0, 1.0, -1.0, f32::MIN, f32::MAX, 0.123456789];
        
        let serialized = SqliteStorage::serialize_embedding(&embedding);
        let deserialized = SqliteStorage::deserialize_embedding(&serialized).unwrap();
        
        assert_eq!(embedding.len(), deserialized.len());
        for (original, restored) in embedding.iter().zip(deserialized.iter()) {
            assert_eq!(original, restored);
        }
    }
    
    #[tokio::test]
    async fn test_deserialize_invalid_bytes() {
        // Test with invalid byte length (not multiple of 4)
        let invalid_bytes = vec![0u8, 1, 2]; // Only 3 bytes
        let result = SqliteStorage::deserialize_embedding(&invalid_bytes);
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::SerializationError(_)));
    }
    
    #[tokio::test]
    async fn test_deserialize_empty_bytes() {
        // Empty bytes should produce empty embedding
        let empty_bytes: Vec<u8> = vec![];
        let result = SqliteStorage::deserialize_embedding(&empty_bytes).unwrap();
        
        assert_eq!(result.len(), 0);
    }
    
    #[tokio::test]
    async fn test_concurrent_reads() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        // Insert a paper
        let embedding = create_test_embedding(128);
        let paper = create_test_paper("Concurrent Test", 2023, embedding);
        let id = storage.insert_paper(&paper).await.unwrap();
        
        // Perform multiple concurrent reads
        let storage_arc = Arc::new(storage);
        let mut handles = vec![];
        
        for _ in 0..10 {
            let storage_clone = Arc::clone(&storage_arc);
            let handle = tokio::spawn(async move {
                storage_clone.get_paper_by_id(id).await.unwrap()
            });
            handles.push(handle);
        }
        
        // All reads should succeed
        for handle in handles {
            let result = handle.await.unwrap();
            assert_eq!(result.id, Some(id));
        }
    }
    
    #[tokio::test]
    async fn test_large_embedding() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        // Test with a very large embedding (e.g., hypothetical 8K dimension)
        let embedding = create_test_embedding(8192);
        let paper = create_test_paper("Large Embedding Paper", 2023, embedding.clone());
        
        let id = storage.insert_paper(&paper).await.unwrap();
        let retrieved = storage.get_paper_by_id(id).await.unwrap();
        
        assert_eq!(retrieved.embedding.as_ref().unwrap().len(), 8192);
        assert_eq!(retrieved.embedding.unwrap(), embedding);
    }
    
    #[tokio::test]
    async fn test_authors_serialization() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        // Create paper with various author configurations
        let embedding = create_test_embedding(128);
        let mut paper = create_test_paper("Multi-Author Paper", 2023, embedding);
        
        paper.authors = vec![
            Author {
                name: "Alice".to_string(),
                affiliation: Some("Harvard".to_string()),
            },
            Author {
                name: "Bob".to_string(),
                affiliation: None, // No affiliation
            },
            Author {
                name: "Charlie O'Brien".to_string(), // Special characters
                affiliation: Some("MIT & Stanford".to_string()),
            },
        ];
        
        let id = storage.insert_paper(&paper).await.unwrap();
        let retrieved = storage.get_paper_by_id(id).await.unwrap();
        
        assert_eq!(retrieved.authors.len(), 3);
        assert_eq!(retrieved.authors[0].name, "Alice");
        assert_eq!(retrieved.authors[0].affiliation, Some("Harvard".to_string()));
        assert_eq!(retrieved.authors[1].name, "Bob");
        assert_eq!(retrieved.authors[1].affiliation, None);
        assert_eq!(retrieved.authors[2].name, "Charlie O'Brien");
    }
    
    #[tokio::test]
    async fn test_special_characters_in_text() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let embedding = create_test_embedding(128);
        let mut paper = create_test_paper("Test", 2023, embedding);
        
        // Test with special characters, unicode, etc.
        paper.title = "Testing SQL Injection'; DROP TABLE papers; --".to_string();
        paper.abstract_text = "Unicode test: 你好世界 🚀 Émilie's résumé".to_string();
        
        let id = storage.insert_paper(&paper).await.unwrap();
        let retrieved = storage.get_paper_by_id(id).await.unwrap();
        
        // Title should be normalized (lowercase)
        assert!(retrieved.title.contains("sql injection"));
        assert_eq!(retrieved.abstract_text, "Unicode test: 你好世界 🚀 Émilie's résumé");
        
        // Verify database wasn't compromised
        let count = storage.count_papers().await.unwrap();
        assert_eq!(count, 1);
    }
    
    #[tokio::test]
    async fn test_operations_without_initialization() {
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap().to_string();
        let storage = SqliteStorage::new(db_path);
        
        // Operations should fail without initialization
        let result = storage.get_config().await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StorageError::ConnectionError(_)));
    }
    
    #[tokio::test]
    async fn test_persistence_across_instances() {
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap().to_string();
        
        // First instance: insert data
        {
            let mut storage1 = SqliteStorage::new(db_path.clone());
            storage1.initialize().await.unwrap();
            
            let config = EmbeddingConfig {
                model_name: "test-model".to_string(),
                dimension: 128,
            };
            storage1.store_config(&config).await.unwrap();
            
            let embedding = create_test_embedding(128);
            let paper = create_test_paper("Persistent Paper", 2023, embedding);
            storage1.insert_paper(&paper).await.unwrap();
        }
        
        // Second instance: verify data persisted
        {
            let mut storage2 = SqliteStorage::new(db_path);
            storage2.initialize().await.unwrap();
            
            let config = storage2.get_config().await.unwrap().unwrap();
            assert_eq!(config.model_name, "test-model");
            
            let count = storage2.count_papers().await.unwrap();
            assert_eq!(count, 1);
            
            let papers = storage2.get_all_papers(None).await.unwrap();
            assert_eq!(papers[0].title, "persistent paper");
        }
    }
    
    #[tokio::test]
    async fn test_year_edge_cases() {
        let (mut storage, _temp_file) = create_test_storage().await;
        
        let embedding = create_test_embedding(128);
        
        // Test with edge case years
        let paper1 = create_test_paper("Ancient Paper", 1900, embedding.clone());
        let paper2 = create_test_paper("Future Paper", 2100, embedding.clone());
        let paper3 = create_test_paper("Negative Year", -100, embedding.clone());
        
        storage.insert_paper(&paper1).await.unwrap();
        storage.insert_paper(&paper2).await.unwrap();
        storage.insert_paper(&paper3).await.unwrap();
        
        // Verify all were inserted
        let count = storage.count_papers().await.unwrap();
        assert_eq!(count, 3);
        
        // Test filtering with edge years
        let year_range = Some(YearRange::new(1900, 2100));
        let papers = storage.get_all_papers(year_range).await.unwrap();
        assert_eq!(papers.len(), 2); // Ancient and Future, not Negative
    }
}
