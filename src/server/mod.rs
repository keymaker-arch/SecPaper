//! MCP server module.
//!
//! This module implements the Model Context Protocol (MCP) server that exposes
//! the paper search functionality. It handles incoming requests, coordinates
//! with the search engine, and formats responses.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::models::SearchResult;
use crate::storage::YearRange;

/// Errors that can occur during MCP server operations.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Invalid request parameters
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    /// Search execution failed
    #[error("Search failed: {0}")]
    SearchError(String),
    
    /// Server initialization error
    #[error("Initialization error: {0}")]
    InitializationError(String),
    
    /// Other unexpected errors
    #[error("Server error: {0}")]
    Other(String),
}

/// Result type for server operations.
pub type ServerResult<T> = Result<T, ServerError>;

/// Request payload for the search_papers endpoint.
///
/// This struct represents the JSON payload sent by clients when requesting
/// a paper search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPapersRequest {
    /// The search query text
    pub query: String,
    
    /// Number of results to return (default: 10)
    #[serde(default = "default_paper_count")]
    pub paper_count: usize,
    
    /// Optional publication year range filter [start_year, end_year] (inclusive)
    #[serde(default)]
    pub publish_year_range: Option<[i32; 2]>,
}

impl SearchPapersRequest {
    /// Convert the year range array to a YearRange struct.
    ///
    /// # Returns
    /// `Some(YearRange)` if publish_year_range is set, `None` otherwise
    pub fn to_year_range(&self) -> Option<YearRange> {
        self.publish_year_range
            .map(|[start, end]| YearRange::new(start, end))
    }
}

/// Default value for paper_count field.
fn default_paper_count() -> usize {
    10
}

/// Response payload for the search_papers endpoint.
///
/// This struct represents the JSON response sent to clients after processing
/// a search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPapersResponse {
    /// List of search results, sorted by relevance (highest score first)
    pub results: Vec<SearchResultDto>,
    
    /// Total number of results returned
    pub count: usize,
    
    /// The original query text
    pub query: String,
}

/// Data transfer object for a single search result.
///
/// This is a simplified version of `SearchResult` optimized for JSON serialization
/// and client consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultDto {
    /// Paper title
    pub title: String,
    
    /// Authors with their affiliations (JSON string for flexibility)
    pub authors: String,
    
    /// Paper abstract
    pub abstract_text: String,
    
    /// Publication year
    pub publish_year: i32,
    
    /// Relevance level as a string (e.g., "IDENTICAL", "HIGHLY_SIMILAR")
    pub relevance: String,
    
    /// Cosine similarity score (0.0 to 1.0)
    pub score: f32,
}

impl From<SearchResult> for SearchResultDto {
    fn from(result: SearchResult) -> Self {
        Self {
            title: result.paper.title,
            authors: serde_json::to_string(&result.paper.authors)
                .unwrap_or_else(|_| "[]".to_string()),
            abstract_text: result.paper.abstract_text,
            publish_year: result.paper.publish_year,
            relevance: format!("{:?}", result.relevance).to_uppercase(),
            score: result.score,
        }
    }
}

/// MCP server configuration.
///
/// This struct holds the configuration needed to run the MCP server,
/// including network settings and resource limits.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server host address
    pub host: String,
    
    /// Server port
    pub port: u16,
    
    /// Maximum number of concurrent requests
    pub max_concurrent_requests: usize,
    
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            max_concurrent_requests: 100,
            request_timeout_secs: 30,
        }
    }
}

/// MCP server instance.
///
/// This struct represents the running MCP server and coordinates all
/// request handling.
pub struct McpServer {
    /// Server configuration
    config: ServerConfig,
    
    // TODO: Add search engine and other dependencies when implementing
}

impl McpServer {
    /// Create a new MCP server instance.
    ///
    /// # Arguments
    /// * `config` - Server configuration
    pub fn new(config: ServerConfig) -> Self {
        Self { config }
    }
    
    /// Initialize the server and prepare for handling requests.
    ///
    /// This method should validate configuration, initialize connections,
    /// and prepare all necessary resources.
    ///
    /// # Errors
    /// Returns `ServerError` if initialization fails
    pub async fn initialize(&mut self) -> ServerResult<()> {
        // TODO: Implement server initialization
        unimplemented!("Server initialization not yet implemented")
    }
    
    /// Start the server and begin handling requests.
    ///
    /// This method blocks until the server is shut down.
    ///
    /// # Errors
    /// Returns `ServerError` if the server fails to start or encounters
    /// a fatal error during execution
    pub async fn run(&self) -> ServerResult<()> {
        // TODO: Implement server run loop
        unimplemented!("Server run loop not yet implemented")
    }
    
    /// Handle a search_papers request.
    ///
    /// # Arguments
    /// * `request` - The search request parameters
    ///
    /// # Returns
    /// A response containing the search results
    ///
    /// # Errors
    /// Returns `ServerError` if the request is invalid or the search fails
    async fn handle_search_papers(
        &self,
        request: SearchPapersRequest,
    ) -> ServerResult<SearchPapersResponse> {
        // TODO: Implement search request handling:
        // 1. Validate request parameters
        // 2. Convert to SearchQuery
        // 3. Execute search
        // 4. Convert results to DTOs
        // 5. Build response
        unimplemented!("Search request handling not yet implemented")
    }
}
