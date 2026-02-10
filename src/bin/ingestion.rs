//! Ingestion pipeline binary entry point.
//!
//! This binary runs the offline ingestion pipeline to process paper metadata,
//! generate embeddings, and build the searchable database.

use mcp_paper_search::{
    embedding::openai::OpenAIEmbedding,
    ingestion::IngestionPipeline,
    provider::{json::JsonFileProvider, PaperProvider},
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
    let mut pipeline = IngestionPipeline::new(embedding_provider, storage, None);
    
    // Initialize the pipeline
    println!("Initializing pipeline...");
    pipeline.initialize().await?;
    
    // TODO: Parse input file path from command-line args
    let input_file = PathBuf::from("papers.json");
    
    // Create provider and load papers
    println!("Loading papers from {:?}...", input_file);
    let provider = JsonFileProvider::new(input_file);
    let papers = provider.fetch_papers().await?;
    println!("Loaded {} papers", papers.len());
    
    // Process papers in batches
    println!("Processing papers...");
    let stats = pipeline.ingest_batch(&papers).await?;
    
    // Display statistics
    println!("\nIngestion completed:");
    println!("  Total processed: {}", stats.total_processed);
    println!("  Inserted: {}", stats.inserted);
    println!("  Duplicates skipped: {}", stats.duplicates_skipped);
    println!("  Failed: {}", stats.failed);
    
    println!("\nIngestion pipeline completed successfully");
    
    Ok(())
}
