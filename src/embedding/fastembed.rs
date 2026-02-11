//! FastEmbed embedding provider implementation.
//!
//! This module provides an implementation of the `EmbeddingProvider` trait
//! using the fastembed library for local embedding generation.
//!
//! FastEmbed allows running embedding models locally without requiring API calls,
//! which can be faster and more cost-effective for batch processing.

use super::{EmbeddingError, EmbeddingProvider, EmbeddingResult};
use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// FastEmbed embedding provider configuration.
///
/// This struct holds the configuration and model instance for generating
/// embeddings using the fastembed library.
#[derive(Clone)]
pub struct FastEmbedProvider {
    /// The embedding model instance (wrapped in Arc<Mutex> for thread-safety)
    model: Arc<Mutex<TextEmbedding>>,
    
    /// Model identifier
    model_name: String,
    
    /// Expected dimension of the embedding vectors
    embedding_dimension: usize,
}

impl FastEmbedProvider {
    /// Create a new FastEmbed embedding provider.
    ///
    /// # Arguments
    /// * `model` - Optional model to use (defaults to AllMiniLML6V2)
    /// * `cache_dir` - Optional cache directory for model files
    ///
    /// # Returns
    /// A new `FastEmbedProvider` instance
    ///
    /// # Errors
    /// Returns `EmbeddingError` if model initialization fails
    pub fn new(
        model: Option<EmbeddingModel>,
        cache_dir: Option<String>,
    ) -> EmbeddingResult<Self> {
        let model_type = model.unwrap_or(EmbeddingModel::AllMiniLML6V2);
        let model_name = format!("{:?}", model_type);
        
        // Determine embedding dimension based on model type
        let embedding_dimension = match model_type {
            EmbeddingModel::AllMiniLML6V2 => 384,
            EmbeddingModel::BGESmallENV15 => 384,
            EmbeddingModel::BGEBaseENV15 => 768,
            EmbeddingModel::BGELargeENV15 => 1024,
            EmbeddingModel::NomicEmbedTextV1 => 768,
            EmbeddingModel::NomicEmbedTextV15 => 768,
            EmbeddingModel::ParaphraseMLMiniLML12V2 => 384,
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => 768,
            _ => 384, // Default fallback
        };
        
        // Initialize the model with optional cache directory
        let mut init_options = InitOptions::new(model_type);
        if let Some(dir) = cache_dir {
            init_options = init_options.with_cache_dir(PathBuf::from(dir));
        }
        
        let text_embedding = TextEmbedding::try_new(init_options)
            .map_err(|e| EmbeddingError::ConfigError(format!("Failed to initialize FastEmbed model: {}", e)))?;
        
        Ok(Self {
            model: Arc::new(Mutex::new(text_embedding)),
            model_name,
            embedding_dimension,
        })
    }
    
    /// Create a new FastEmbed provider with default settings.
    ///
    /// Uses AllMiniLML6V2 model with default cache directory.
    ///
    /// # Returns
    /// A new `FastEmbedProvider` instance
    ///
    /// # Errors
    /// Returns `EmbeddingError` if model initialization fails
    pub fn default() -> EmbeddingResult<Self> {
        Self::new(None, None)
    }
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    async fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(EmbeddingError::InvalidInput("Text cannot be empty".to_string()));
        }
        
        // Lock the model for embedding generation
        let mut model = self.model.lock().await;
        
        // Generate embedding (fastembed expects a vector of texts)
        let embeddings = model
            .embed(vec![text.to_string()], None)
            .map_err(|e| EmbeddingError::Other(format!("Embedding generation failed: {}", e)))?;
        
        // Extract the first (and only) embedding
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::Other("No embedding generated".to_string()))
    }
    
    async fn embed_batch(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        
        // Validate inputs
        for text in texts {
            if text.trim().is_empty() {
                return Err(EmbeddingError::InvalidInput("All texts must be non-empty".to_string()));
            }
        }
        
        // Lock the model for embedding generation
        let mut model = self.model.lock().await;
        
        // Convert &[&str] to Vec<String> for fastembed
        let text_strings: Vec<String> = texts.iter().map(|&s| s.to_string()).collect();
        
        // Generate embeddings for all texts
        let embeddings = model
            .embed(text_strings, None)
            .map_err(|e| EmbeddingError::Other(format!("Batch embedding generation failed: {}", e)))?;
        
        Ok(embeddings)
    }
    
    fn dimension(&self) -> usize {
        self.embedding_dimension
    }
    
    fn model_name(&self) -> &str {
        &self.model_name
    }
}

// Implementing Debug manually to avoid issues with TextEmbedding not implementing Debug
impl std::fmt::Debug for FastEmbedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedProvider")
            .field("model_name", &self.model_name)
            .field("embedding_dimension", &self.embedding_dimension)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastembed::EmbeddingModel;

    /// Helper function to create a test provider with default settings
    fn create_test_provider() -> FastEmbedProvider {
        FastEmbedProvider::default()
            .expect("Failed to create default FastEmbedProvider")
    }

    #[test]
    fn test_provider_creation_default() {
        let provider = FastEmbedProvider::default();
        assert!(provider.is_ok(), "Default provider creation should succeed");
        
        let provider = provider.unwrap();
        assert_eq!(provider.dimension(), 384, "Default model should have 384 dimensions");
        assert!(provider.model_name().contains("AllMiniLML6V2"), 
                "Default model should be AllMiniLML6V2");
    }

    #[test]
    fn test_provider_creation_with_model() {
        let provider = FastEmbedProvider::new(
            Some(EmbeddingModel::BGEBaseENV15),
            None,
        );
        assert!(provider.is_ok(), "Provider creation with specific model should succeed");
        
        let provider = provider.unwrap();
        assert_eq!(provider.dimension(), 768, "BGEBaseENV15 should have 768 dimensions");
        assert!(provider.model_name().contains("BGEBaseENV15"),
                "Model name should contain BGEBaseENV15");
    }

    #[test]
    fn test_provider_creation_with_cache_dir() {
        let temp_dir = std::env::temp_dir().join("fastembed_test_cache");
        let provider = FastEmbedProvider::new(
            Some(EmbeddingModel::AllMiniLML6V2),
            Some(temp_dir.to_string_lossy().to_string()),
        );
        assert!(provider.is_ok(), "Provider creation with cache dir should succeed");
    }

    #[tokio::test]
    async fn test_embed_single_text() {
        let provider = create_test_provider();
        let text = "This is a test sentence for embedding generation.";
        
        let result = provider.embed(text).await;
        assert!(result.is_ok(), "Embedding generation should succeed");
        
        let embedding = result.unwrap();
        assert_eq!(
            embedding.len(),
            provider.dimension(),
            "Embedding should have correct dimension"
        );
        
        // Check that embedding contains valid values
        assert!(embedding.iter().all(|&x| x.is_finite()), 
                "All embedding values should be finite");
    }

    #[tokio::test]
    async fn test_embed_empty_text() {
        let provider = create_test_provider();
        
        let result = provider.embed("").await;
        assert!(result.is_err(), "Embedding empty text should fail");
        
        if let Err(EmbeddingError::InvalidInput(msg)) = result {
            assert!(msg.contains("empty"), "Error message should mention empty text");
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[tokio::test]
    async fn test_embed_whitespace_only() {
        let provider = create_test_provider();
        
        let result = provider.embed("   \n\t  ").await;
        assert!(result.is_err(), "Embedding whitespace-only text should fail");
    }

    #[tokio::test]
    async fn test_embed_batch_multiple_texts() {
        let provider = create_test_provider();
        let texts = vec![
            "First test sentence.",
            "Second test sentence with different content.",
            "Third sentence about embeddings.",
        ];
        
        let result = provider.embed_batch(&texts).await;
        assert!(result.is_ok(), "Batch embedding should succeed");
        
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), texts.len(), 
                   "Should generate embedding for each text");
        
        // Verify all embeddings have correct dimension
        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(
                embedding.len(),
                provider.dimension(),
                "Embedding {} should have correct dimension",
                i
            );
            assert!(embedding.iter().all(|&x| x.is_finite()),
                    "All values in embedding {} should be finite", i);
        }
        
        // Verify embeddings are different for different texts
        assert_ne!(embeddings[0], embeddings[1],
                   "Different texts should produce different embeddings");
    }

    #[tokio::test]
    async fn test_embed_batch_empty_vec() {
        let provider = create_test_provider();
        let texts: Vec<&str> = vec![];
        
        let result = provider.embed_batch(&texts).await;
        assert!(result.is_ok(), "Empty batch should succeed");
        assert_eq!(result.unwrap().len(), 0, "Empty batch should return empty vec");
    }

    #[tokio::test]
    async fn test_embed_batch_with_empty_text() {
        let provider = create_test_provider();
        let texts = vec!["Valid text", "", "Another valid text"];
        
        let result = provider.embed_batch(&texts).await;
        assert!(result.is_err(), "Batch with empty text should fail");
        
        if let Err(EmbeddingError::InvalidInput(msg)) = result {
            assert!(msg.contains("non-empty"), 
                    "Error message should mention non-empty requirement");
        } else {
            panic!("Expected InvalidInput error");
        }
    }

    #[tokio::test]
    async fn test_embed_batch_single_text() {
        let provider = create_test_provider();
        let texts = vec!["Single text in batch"];
        
        let result = provider.embed_batch(&texts).await;
        assert!(result.is_ok(), "Single text batch should succeed");
        
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1, "Should return one embedding");
        assert_eq!(embeddings[0].len(), provider.dimension());
    }

    #[tokio::test]
    async fn test_embed_consistency() {
        // Same text should produce same embedding
        let provider = create_test_provider();
        let text = "Consistency test text";
        
        let embedding1 = provider.embed(text).await.unwrap();
        let embedding2 = provider.embed(text).await.unwrap();
        
        assert_eq!(embedding1, embedding2,
                   "Same text should produce identical embeddings");
    }

    #[tokio::test]
    async fn test_embed_batch_consistency_with_single_embed() {
        // Batch embedding should produce same result as single embedding
        let provider = create_test_provider();
        let text = "Test text for consistency";
        
        let single_embedding = provider.embed(text).await.unwrap();
        let batch_embeddings = provider.embed_batch(&vec![text]).await.unwrap();
        
        assert_eq!(single_embedding, batch_embeddings[0],
                   "Single embed and batch embed should produce same result");
    }

    #[tokio::test]
    async fn test_embed_long_text() {
        let provider = create_test_provider();
        // Create a longer text to test handling
        let long_text = "This is a longer piece of text. ".repeat(50);
        
        let result = provider.embed(&long_text).await;
        assert!(result.is_ok(), "Embedding long text should succeed");
        
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), provider.dimension());
    }

    #[tokio::test]
    async fn test_embed_special_characters() {
        let provider = create_test_provider();
        let text = "Text with special chars: @#$%^&*() and unicode: 你好 мир 🌍";
        
        let result = provider.embed(text).await;
        assert!(result.is_ok(), "Embedding text with special characters should succeed");
        
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), provider.dimension());
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_concurrent_embeddings() {
        // Test thread-safety by running concurrent embeddings
        let provider = create_test_provider();
        let provider = Arc::new(provider);
        
        let mut handles = vec![];
        
        for i in 0..5 {
            let provider_clone = Arc::clone(&provider);
            let handle = tokio::spawn(async move {
                let text = format!("Concurrent test text {}", i);
                provider_clone.embed(&text).await
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        for handle in handles {
            let result = handle.await.expect("Task should complete");
            assert!(result.is_ok(), "Concurrent embedding should succeed");
        }
    }

    #[test]
    fn test_debug_implementation() {
        let provider = create_test_provider();
        let debug_str = format!("{:?}", provider);
        
        assert!(debug_str.contains("FastEmbedProvider"),
                "Debug output should contain struct name");
        assert!(debug_str.contains("model_name"),
                "Debug output should contain model_name field");
        assert!(debug_str.contains("embedding_dimension"),
                "Debug output should contain embedding_dimension field");
    }

    #[test]
    fn test_clone() {
        let provider = create_test_provider();
        let cloned = provider.clone();
        
        assert_eq!(provider.dimension(), cloned.dimension(),
                   "Cloned provider should have same dimension");
        assert_eq!(provider.model_name(), cloned.model_name(),
                   "Cloned provider should have same model name");
    }

    #[tokio::test]
    async fn test_provider_methods() {
        let provider = create_test_provider();
        
        // Test dimension()
        let dim = provider.dimension();
        assert!(dim > 0, "Dimension should be positive");
        
        // Test model_name()
        let name = provider.model_name();
        assert!(!name.is_empty(), "Model name should not be empty");
        
        // Verify these are consistent with actual embeddings
        let embedding = provider.embed("test").await.unwrap();
        assert_eq!(embedding.len(), dim,
                   "Actual embedding dimension should match dimension()");
    }
}
