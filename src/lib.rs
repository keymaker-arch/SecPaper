//! MCP Paper Search - A semantic search engine for research papers.
//!
//! This library provides the core functionality for the MCP paper search system,
//! which allows semantic search over research papers using abstract embeddings.
//!
//! # Architecture
//!
//! The system is organized into several key modules:
//!
//! - **models**: Core data structures (Paper, Author, SearchResult, etc.)
//! - **embedding**: Text embedding generation and normalization
//! - **storage**: Database persistence and retrieval (SQLite-based)
//! - **query**: Search execution and ranking algorithms
//! - **ingestion**: Offline data ingestion pipeline
//! - **server**: MCP server implementation
//!
//! # Workflow
//!
//! ## Offline Ingestion
//!
//! 1. Load paper metadata from input sources
//! 2. Normalize titles for deduplication
//! 3. Generate embeddings for paper abstracts
//! 4. Store papers and embeddings in SQLite database
//!
//! ## Online Search
//!
//! 1. Receive search query from MCP client
//! 2. Normalize and embed the query text
//! 3. Retrieve papers from database (with optional filters)
//! 4. Compute cosine similarity between query and papers
//! 5. Return top-k ranked results
//!
//! # Example
//!
//! ```ignore
//! use mcp_paper_search::{
//!     embedding::openai::OpenAIEmbedding,
//!     storage::sqlite::SqliteStorage,
//!     query::{BruteForceSearchEngine, SearchQuery},
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Set up components
//!     let embedding = OpenAIEmbedding::new(api_key, None);
//!     let storage = SqliteStorage::new("papers.db".to_string());
//!     let engine = BruteForceSearchEngine::new(embedding, storage);
//!     
//!     // Execute search
//!     let query = SearchQuery::new("deep learning".to_string(), Some(10), None);
//!     let results = engine.search(&query).await?;
//!     
//!     // Process results
//!     for result in results {
//!         println!("{}: {}", result.paper.title, result.score);
//!     }
//!     
//!     Ok(())
//! }
//! ```

// Public modules
pub mod embedding;
pub mod ingestion;
pub mod models;
pub mod query;
pub mod server;
pub mod storage;

// Re-export commonly used types at the crate root
pub use models::{Author, Paper, SearchResult, RelevanceLevel, EmbeddingConfig};
pub use embedding::EmbeddingProvider;
pub use storage::{PaperStorage, YearRange};
pub use query::{SearchEngine, SearchQuery};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default embedding model name
pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-3-small";

/// Default embedding dimension for text-embedding-3-small
pub const DEFAULT_EMBEDDING_DIMENSION: usize = 1536;
