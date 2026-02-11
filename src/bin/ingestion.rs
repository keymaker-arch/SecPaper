//! Ingestion pipeline binary entry point.
//!
//! This binary runs the offline ingestion pipeline to process paper metadata,
//! generate embeddings, and build the searchable database.

use mcp_paper_search::{
    embedding::openai::OpenAIEmbedding,
    ingestion::IngestionPipeline,
    provider::{json::JsonFilePaperProvider, PaperProvider},
    storage::sqlite::SqliteStorage,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Parse command-line arguments
    // TODO: Load configuration from file
    // TODO: Set up logging
    // TODO: Read input data (JSON, CSV, etc.)
    
    println!("Starting paper ingestion pipeline...");
    
    // TODO: Get API key from environment or config
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");
    
    // Initialize components
    let embedding_provider = OpenAIEmbedding::new(api_key, None);
    let storage = SqliteStorage::new("papers.db".to_string());
    
    // Initialize the pipeline with new storage
    println!("Initializing pipeline...");
    let mut pipeline = IngestionPipeline::initialize_new(embedding_provider, storage, None).await?;
    
    // TODO: Parse input file path from command-line args
    let input_file = PathBuf::from("papers.json");
    
    // Create provider
    println!("Loading papers from {:?}...", input_file);
    let provider = JsonFilePaperProvider::from_file(input_file).await?;
    
    // Get paper count for progress tracking
    let paper_count = provider.count_papers().await?;
    println!("Found {} papers from {}", paper_count, provider.name());
    
    // Process papers directly from provider
    println!("Processing papers...");
    let stats = pipeline.ingest_from_provider(&provider).await?;
    
    // Display statistics
    println!("\nIngestion completed:");
    println!("  Total processed: {}", stats.total_processed);
    println!("  Inserted: {}", stats.inserted);
    println!("  Duplicates skipped: {}", stats.duplicates_skipped);
    println!("  Failed: {}", stats.failed);
    
    println!("\nIngestion pipeline completed successfully");
    
    Ok(())
}
