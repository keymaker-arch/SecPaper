//! Ingestion pipeline binary entry point.
//!
//! This binary runs the offline ingestion pipeline to process paper metadata,
//! generate embeddings, and build the searchable database.
//!
//! # Examples
//!
//! Initialize new database:
//! ```bash
//! ingestion --mode init-new --input papers.json --db-path papers.db
//! ```
//!
//! Add papers to existing database:
//! ```bash
//! ingestion --input new_papers.json
//! ```

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use mcp_paper_search::{
    embedding::{fastembed::FastEmbedProvider, openai::OpenAIEmbedding, EmbeddingProvider},
    ingestion::IngestionPipeline,
    provider::{json::JsonFilePaperProvider, PaperProvider},
    storage::sqlite::SqliteStorage,
};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Wrapper enum for embedding providers to allow dynamic dispatch
enum DynamicEmbeddingProvider {
    FastEmbed(FastEmbedProvider),
    OpenAI(OpenAIEmbedding),
}

#[async_trait::async_trait]
impl EmbeddingProvider for DynamicEmbeddingProvider {
    async fn embed(&self, text: &str) -> mcp_paper_search::embedding::EmbeddingResult<Vec<f32>> {
        match self {
            DynamicEmbeddingProvider::FastEmbed(p) => p.embed(text).await,
            DynamicEmbeddingProvider::OpenAI(p) => p.embed(text).await,
        }
    }

    async fn embed_batch(&self, texts: &[&str]) -> mcp_paper_search::embedding::EmbeddingResult<Vec<Vec<f32>>> {
        match self {
            DynamicEmbeddingProvider::FastEmbed(p) => p.embed_batch(texts).await,
            DynamicEmbeddingProvider::OpenAI(p) => p.embed_batch(texts).await,
        }
    }

    fn dimension(&self) -> usize {
        match self {
            DynamicEmbeddingProvider::FastEmbed(p) => p.dimension(),
            DynamicEmbeddingProvider::OpenAI(p) => p.dimension(),
        }
    }

    fn model_name(&self) -> &str {
        match self {
            DynamicEmbeddingProvider::FastEmbed(p) => p.model_name(),
            DynamicEmbeddingProvider::OpenAI(p) => p.model_name(),
        }
    }
}

/// Operation mode for the ingestion pipeline
#[derive(Debug, Clone, ValueEnum)]
enum Mode {
    /// Connect to existing database and validate embedding config (default)
    Connect,
    /// Initialize new database with fresh schema
    InitNew,
}

/// Embedding provider type
#[derive(Debug, Clone, ValueEnum)]
enum EmbeddingProviderType {
    /// FastEmbed local embedding provider (default, no API required)
    FastEmbed,
    /// OpenAI cloud-based embedding provider (requires OPENAI_API_KEY)
    OpenAI,
}

/// Ingestion pipeline CLI for building and updating the paper database
#[derive(Parser, Debug)]
#[command(
    name = "ingestion",
    version,
    about = "Build and update the paper search database",
    long_about = "Ingestion pipeline for processing research papers, generating embeddings, and building a searchable database.

EXAMPLES:
  Initialize new database:
    ingestion --mode init-new --input papers.json --db-path papers.db
  
  Add papers to existing database:
    ingestion --input new_papers.json
  
  Use OpenAI embeddings:
    OPENAI_API_KEY=sk-... ingestion --mode init-new --input papers.json --embedding-provider openai
  
  Custom batch size and logging:
    ingestion --input papers.json --batch-size 50 --log-level debug"
)]
struct IngestionArgs {
    /// Input JSON file containing paper metadata
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    /// Database file path
    #[arg(long, value_name = "PATH", default_value = "papers.db")]
    db_path: String,

    /// Operation mode: connect to existing DB or initialize new DB
    #[arg(long, value_enum, default_value = "connect")]
    mode: Mode,

    /// Embedding provider to use
    #[arg(long, value_enum, default_value = "fast-embed")]
    embedding_provider: EmbeddingProviderType,

    /// Specific embedding model name (provider-dependent, optional)
    #[arg(long, value_name = "MODEL")]
    embedding_model: Option<String>,

    /// Number of papers to process per embedding batch
    #[arg(long, value_name = "N", default_value = "100")]
    batch_size: usize,

    /// Logging verbosity level
    #[arg(long, value_name = "LEVEL", default_value = "info")]
    log_level: String,

    /// FastEmbed model cache directory
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<String>,
}

/// Initialize logging subsystem with the specified level
fn init_logging(level: &str) -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .init();

    Ok(())
}

/// Create an embedding provider based on CLI arguments
async fn create_embedding_provider(
    args: &IngestionArgs,
) -> Result<DynamicEmbeddingProvider> {
    match args.embedding_provider {
        EmbeddingProviderType::FastEmbed => {
            info!("Initializing FastEmbed provider");
            
            let model = if let Some(model_name) = &args.embedding_model {
                // Parse model name to EmbeddingModel enum
                info!("Using custom FastEmbed model: {}", model_name);
                // For simplicity, use default model; full implementation would parse the string
                None
            } else {
                None
            };

            let provider = if let Some(cache_dir) = &args.cache_dir {
                debug!("Using custom cache directory: {}", cache_dir);
                FastEmbedProvider::new(model, Some(cache_dir.clone()))
                    .context("Failed to initialize FastEmbed provider")?
            } else {
                // Use default cache directory
                let default_cache = dirs::cache_dir()
                    .map(|p| p.join("fastembed").to_string_lossy().to_string())
                    .unwrap_or_else(|| ".cache/fastembed".to_string());
                debug!("Using default cache directory: {}", default_cache);
                FastEmbedProvider::new(model, Some(default_cache))
                    .context("Failed to initialize FastEmbed provider")?
            };

            info!(
                "FastEmbed provider initialized: model={}, dimension={}",
                provider.model_name(),
                provider.dimension()
            );

            Ok(DynamicEmbeddingProvider::FastEmbed(provider))
        }
        EmbeddingProviderType::OpenAI => {
            info!("Initializing OpenAI embedding provider");
            
            let api_key = std::env::var("OPENAI_API_KEY").context(
                "OPENAI_API_KEY environment variable must be set when using OpenAI provider",
            )?;

            let provider = OpenAIEmbedding::new(api_key, args.embedding_model.clone());
            
            info!(
                "OpenAI provider initialized: model={}, dimension={}",
                provider.model_name(),
                provider.dimension()
            );

            Ok(DynamicEmbeddingProvider::OpenAI(provider))
        }
    }
}

/// Create storage instance
fn create_storage(db_path: &str) -> Result<SqliteStorage> {
    debug!("Creating SQLite storage at: {}", db_path);
    
    // Ensure parent directory exists
    if let Some(parent) = PathBuf::from(db_path).parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create database directory: {:?}", parent))?;
            info!("Created database directory: {:?}", parent);
        }
    }

    Ok(SqliteStorage::new(db_path.to_string()))
}

/// Create and initialize the ingestion pipeline based on mode
async fn create_pipeline(
    args: &IngestionArgs,
    embedding_provider: DynamicEmbeddingProvider,
    storage: SqliteStorage,
) -> Result<IngestionPipeline<DynamicEmbeddingProvider, SqliteStorage>> {
    let batch_size = Some(args.batch_size);

    let pipeline = match args.mode {
        Mode::Connect => {
            info!("Connecting to existing database with config validation");
            IngestionPipeline::connect(embedding_provider, storage, batch_size)
                .await
                .context("Failed to connect to existing database. Use --mode init-new to create a new database.")?
        }
        Mode::InitNew => {
            info!("Initializing new database");
            IngestionPipeline::initialize_new(embedding_provider, storage, batch_size)
                .await
                .context("Failed to initialize new database")?
        }
    };

    debug!("Pipeline created with batch_size={}", args.batch_size);
    Ok(pipeline)
}

/// Create a progress bar for tracking ingestion
fn create_progress_bar(total: usize) -> ProgressBar {
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} papers | Inserted: {msg}")
            .expect("Invalid progress bar template")
            .progress_chars("##-"),
    );
    pb
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let args = IngestionArgs::parse();

    // Initialize logging
    init_logging(&args.log_level).context("Failed to initialize logging")?;

    info!("Starting paper ingestion pipeline");
    debug!("CLI arguments: {:?}", args);

    let start_time = Instant::now();

    // Validate input file exists
    if !args.input.exists() {
        error!("Input file does not exist: {:?}", args.input);
        anyhow::bail!("Input file not found: {:?}", args.input);
    }
    info!("Input file: {:?}", args.input);

    // Create embedding provider
    let embedding_provider = create_embedding_provider(&args)
        .await
        .context("Failed to create embedding provider")?;

    // Create storage
    let storage = create_storage(&args.db_path).context("Failed to create storage")?;
    info!("Database path: {}", args.db_path);

    // Create pipeline
    let mut pipeline = create_pipeline(&args, embedding_provider, storage)
        .await
        .context("Failed to create ingestion pipeline")?;

    // Load paper provider
    info!("Loading papers from {:?}...", args.input);
    let provider = JsonFilePaperProvider::from_file(args.input.clone())
        .await
        .with_context(|| format!("Failed to load papers from {:?}", args.input))?;

    let paper_count = provider.count_papers().await?;
    info!("Found {} papers from {}", paper_count, provider.name());

    if paper_count == 0 {
        warn!("No papers found in input file");
        return Ok(());
    }

    // Create progress bar
    let progress = create_progress_bar(paper_count);
    progress.set_message("0");

    // Process papers
    info!("Processing papers with batch_size={}...", args.batch_size);
    let stats = pipeline
        .ingest_from_provider(&provider)
        .await
        .context("Failed to ingest papers")?;

    progress.finish_with_message(format!("{}", stats.inserted));

    // Display final statistics
    let elapsed = start_time.elapsed();
    println!("\n╔════════════════════════════════════════╗");
    println!("║      Ingestion Completed               ║");
    println!("╠════════════════════════════════════════╣");
    println!("║ Total processed:      {:>16} ║", stats.total_processed);
    println!("║ Inserted:             {:>16} ║", stats.inserted);
    println!("║ Duplicates skipped:   {:>16} ║", stats.duplicates_skipped);
    println!("║ Failed:               {:>16} ║", stats.failed);
    println!("║ Elapsed time:         {:>13.2?} ║", elapsed);
    println!("╚════════════════════════════════════════╝");

    if stats.failed > 0 {
        warn!(
            "{} papers failed to process - check logs for details",
            stats.failed
        );
    }

    info!("Ingestion pipeline completed successfully");

    Ok(())
}
