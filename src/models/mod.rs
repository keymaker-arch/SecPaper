//! Core data models for the MCP paper search system.
//!
//! This module contains the fundamental data structures used across the application,
//! including paper metadata, author information, and search results.

use serde::{Deserialize, Serialize};

/// Represents a single author with their affiliation information.
///
/// Authors are stored as part of paper metadata and include both the author's name
/// and their institutional affiliation at the time of publication.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Author {
    /// Full name of the author
    pub name: String,
    
    /// Institutional affiliation (e.g., university, research lab)
    pub affiliation: Option<String>,
}

/// Core metadata for a research paper.
///
/// This struct represents all the essential information about a paper that is
/// stored in the database and returned in search results. The embedding field
/// contains the vector representation of the paper's abstract for semantic search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Paper {
    /// Unique identifier (database primary key)
    pub id: Option<i64>,
    
    /// Paper title
    pub title: String,
    
    /// List of authors with their affiliations
    pub authors: Vec<Author>,
    
    /// Abstract text
    pub abstract_text: String,
    
    /// Year of publication (used for filtering)
    pub publish_year: i32,
    
    /// Vector embedding of the abstract (float32 array)
    /// Dimension depends on the embedding model used (e.g., 1536 for OpenAI text-embedding-3-small)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// Relevance classification for search results.
///
/// Papers are categorized by their semantic similarity to the query,
/// allowing clients to understand the quality of matches.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelevanceLevel {
    /// Extremely high similarity (cosine similarity > 0.95)
    Identical,
    
    /// Very high similarity (cosine similarity > 0.85)
    HighlySimilar,
    
    /// Moderate similarity (cosine similarity > 0.70)
    Similar,
    
    /// Lower similarity but still relevant (cosine similarity > 0.50)
    Relevant,
}

impl RelevanceLevel {
    /// Determine relevance level from a cosine similarity score.
    ///
    /// # Arguments
    /// * `score` - Cosine similarity score between 0.0 and 1.0
    ///
    /// # Returns
    /// The appropriate relevance level for the given score
    pub fn from_score(score: f32) -> Self {
        if score > 0.95 {
            RelevanceLevel::Identical
        } else if score > 0.85 {
            RelevanceLevel::HighlySimilar
        } else if score > 0.70 {
            RelevanceLevel::Similar
        } else {
            RelevanceLevel::Relevant
        }
    }
}

/// A single search result containing paper metadata and relevance information.
///
/// This is the primary output type returned by the search API, combining
/// the paper's metadata with information about how well it matches the query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The paper metadata
    pub paper: Paper,
    
    /// Cosine similarity score (0.0 to 1.0, higher is better)
    pub score: f32,
    
    /// Categorical relevance classification
    pub relevance: RelevanceLevel,
}

impl SearchResult {
    /// Create a new search result from a paper and similarity score.
    ///
    /// # Arguments
    /// * `paper` - The paper metadata
    /// * `score` - Cosine similarity score
    pub fn new(paper: Paper, score: f32) -> Self {
        Self {
            paper,
            score,
            relevance: RelevanceLevel::from_score(score),
        }
    }
}

/// Configuration for the embedding model.
///
/// This configuration is stored alongside the database to ensure consistency
/// between ingestion and query-time embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Name/identifier of the embedding model (e.g., "text-embedding-3-small")
    pub model_name: String,
    
    /// Dimension of the embedding vectors
    pub dimension: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevance_level_from_score() {
        assert_eq!(RelevanceLevel::from_score(0.96), RelevanceLevel::Identical);
        assert_eq!(RelevanceLevel::from_score(0.90), RelevanceLevel::HighlySimilar);
        assert_eq!(RelevanceLevel::from_score(0.75), RelevanceLevel::Similar);
        assert_eq!(RelevanceLevel::from_score(0.60), RelevanceLevel::Relevant);
    }
}
