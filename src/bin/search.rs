//! Search binary entry point.
//!
//! This binary provides a command-line interface for searching papers in a pre-built
//! database. It supports both single-query and interactive REPL modes, with flexible
//! output formatting (table or JSON).
//!
//! # Examples
//!
//! Single query with default settings:
//! ```bash
//! search --db-path papers.db --query "neural networks"
//! ```
//!
//! JSON output with year filter:
//! ```bash
//! search --db-path papers.db --query "transformers" --format json --year-start 2020
//! ```
//!
//! Interactive mode:
//! ```bash
//! search --db-path papers.db --interactive
//! ```

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use comfy_table::{presets::UTF8_FULL, Attribute, Cell, Color, ContentArrangement, Table};
use mcp_paper_search::{
    embedding::{fastembed::FastEmbedProvider, openai::OpenAIEmbedding, EmbeddingProvider},
    models::{RelevanceLevel, SearchResult},
    query::{BruteForceSearchEngine, SearchEngine, SearchQuery},
    storage::{sqlite::SqliteStorage, PaperStorage, YearRange},
};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, error, info};
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

/// Output format for search results
#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    /// Human-friendly table with colored relevance levels
    Table,
    /// Machine-readable JSON format
    Json,
}

/// Search binary CLI for querying the paper database
#[derive(Parser, Debug)]
#[command(
    name = "search",
    version,
    about = "Search papers in the database using semantic similarity",
    long_about = "Query the paper database using semantic search. Supports both single-query \
                  and interactive modes with flexible output formatting.

EXAMPLES:
  Single query:
    search --db-path papers.db --query \"neural networks\"
  
  JSON output with year filter:
    search --db-path papers.db --query \"transformers\" --format json --year-start 2020
  
  Interactive mode:
    search --db-path papers.db --interactive
  
  Top 20 results from recent papers:
    search --db-path papers.db --query \"NLP\" --top-k 20 --year-start 2020"
)]
struct Args {
    /// Database file path
    #[arg(long, value_name = "PATH")]
    db_path: PathBuf,

    /// Search query (required for single-query mode, omitted in interactive mode)
    #[arg(long, value_name = "TEXT", conflicts_with = "interactive")]
    query: Option<String>,

    /// Number of results to return
    #[arg(long, value_name = "N", default_value = "10")]
    top_k: usize,

    /// Filter papers from this year onwards (inclusive)
    #[arg(long, value_name = "YEAR")]
    year_start: Option<i32>,

    /// Filter papers up to this year (inclusive)
    #[arg(long, value_name = "YEAR")]
    year_end: Option<i32>,

    /// Output format
    #[arg(long, value_enum, default_value = "table")]
    format: OutputFormat,

    /// Enable interactive REPL mode
    #[arg(long, short = 'i')]
    interactive: bool,

    /// Logging verbosity level
    #[arg(long, default_value = "warn", value_name = "LEVEL")]
    log_level: String,

    /// FastEmbed model cache directory (only used with FastEmbed provider)
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,
}

/// Setup logging with the specified level
fn setup_logging(log_level: &str) {
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(log_level)),
        )
        .init();
}

/// Auto-detect and instantiate the embedding provider based on database config
async fn create_embedding_provider(
    model_name: &str,
    dimension: usize,
    cache_dir: Option<PathBuf>,
) -> Result<DynamicEmbeddingProvider> {
    info!("Auto-detecting embedding provider for model: {}", model_name);

    // Check if it's an OpenAI model
    if model_name.contains("text-embedding") {
        info!("Detected OpenAI embedding model");
        let api_key = std::env::var("OPENAI_API_KEY").with_context(|| {
            "OPENAI_API_KEY environment variable required for OpenAI embeddings.\n\
             Set it with: export OPENAI_API_KEY=your-api-key"
        })?;

        let provider = OpenAIEmbedding::new(api_key, Some(model_name.to_string()));
        
        // Validate dimension matches
        if provider.dimension() != dimension {
            anyhow::bail!(
                "Dimension mismatch: expected {} from database config, but provider returns {}",
                dimension,
                provider.dimension()
            );
        }

        Ok(DynamicEmbeddingProvider::OpenAI(provider))
    } else {
        // Assume FastEmbed for all other models
        info!("Detected FastEmbed model");
        
        let provider = if let Some(cache_dir) = cache_dir {
            FastEmbedProvider::new(None, Some(cache_dir.to_string_lossy().to_string()))
                .with_context(|| "Failed to create FastEmbed provider with custom cache directory")?
        } else {
            FastEmbedProvider::new(None, None)
                .with_context(|| "Failed to create FastEmbed provider")?
        };

        // Validate dimension matches
        if provider.dimension() != dimension {
            anyhow::bail!(
                "Dimension mismatch: expected {} from database config, but provider returns {}",
                dimension,
                provider.dimension()
            );
        }

        Ok(DynamicEmbeddingProvider::FastEmbed(provider))
    }
}

/// Execute a search query and return results
async fn execute_search<E: EmbeddingProvider, S: PaperStorage>(
    engine: &BruteForceSearchEngine<E, S>,
    query_text: &str,
    top_k: usize,
    year_range: Option<YearRange>,
) -> Result<Vec<SearchResult>> {
    debug!("Executing search for query: {}", query_text);
    
    let query = SearchQuery::new(query_text.to_string(), Some(top_k), year_range);
    
    let results = engine
        .search(&query)
        .await
        .with_context(|| format!("Failed to execute search for query: '{}'", query_text))?;
    
    Ok(results)
}

/// Format results as a pretty table
fn format_results_table(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);

    // Add header
    table.set_header(vec![
        Cell::new("Rank").add_attribute(Attribute::Bold),
        Cell::new("Title").add_attribute(Attribute::Bold),
        Cell::new("Authors").add_attribute(Attribute::Bold),
        Cell::new("Year").add_attribute(Attribute::Bold),
        Cell::new("Relevance").add_attribute(Attribute::Bold),
        Cell::new("Score").add_attribute(Attribute::Bold),
    ]);

    // Add rows
    for (idx, result) in results.iter().enumerate() {
        let authors_str = result
            .paper
            .authors
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        // Truncate long author lists
        let authors_display = if authors_str.len() > 40 {
            format!("{}...", &authors_str[..37])
        } else {
            authors_str
        };

        // Truncate long titles
        let title_display = if result.paper.title.len() > 60 {
            format!("{}...", &result.paper.title[..57])
        } else {
            result.paper.title.clone()
        };

        // Color-code relevance
        let (relevance_str, color) = match result.relevance {
            RelevanceLevel::Identical => ("IDENTICAL", Color::Green),
            RelevanceLevel::HighlySimilar => ("HIGHLY_SIMILAR", Color::Cyan),
            RelevanceLevel::Similar => ("SIMILAR", Color::Yellow),
            RelevanceLevel::Relevant => ("RELEVANT", Color::White),
        };

        table.add_row(vec![
            Cell::new(format!("{}", idx + 1)),
            Cell::new(title_display),
            Cell::new(authors_display),
            Cell::new(result.paper.publish_year),
            Cell::new(relevance_str).fg(color),
            Cell::new(format!("{:.4}", result.score)),
        ]);
    }

    table.to_string()
}

/// Format results as JSON
fn format_results_json(results: &[SearchResult]) -> Result<String> {
    serde_json::to_string_pretty(results)
        .with_context(|| "Failed to serialize results to JSON")
}

/// Display detailed view of a single result
fn display_result_detail(result: &SearchResult, rank: usize) {
    println!("\n{}", "═".repeat(80));
    println!("Rank: {}", rank);
    println!("Title: {}", result.paper.title);
    println!(
        "Authors: {}",
        result
            .paper
            .authors
            .iter()
            .map(|a| {
                if let Some(aff) = &a.affiliation {
                    format!("{} ({})", a.name, aff)
                } else {
                    a.name.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("Year: {}", result.paper.publish_year);
    println!("Relevance: {:?}", result.relevance);
    println!("Score: {:.4}", result.score);
    println!("\nAbstract:\n{}", result.paper.abstract_text);
    println!("{}", "═".repeat(80));
}

/// Run interactive REPL mode
async fn run_interactive<E: EmbeddingProvider + Send + Sync, S: PaperStorage + Send + Sync>(
    engine: BruteForceSearchEngine<E, S>,
    mut top_k: usize,
    mut year_range: Option<YearRange>,
    mut format: OutputFormat,
) -> Result<()> {
    println!("Interactive Paper Search");
    println!("Commands:");
    println!("  <query>         - Search for papers");
    println!("  /top N          - Set number of results to N");
    println!("  /year START END - Filter by year range");
    println!("  /year clear     - Clear year filter");
    println!("  /format table   - Use table output format");
    println!("  /format json    - Use JSON output format");
    println!("  /detail N       - Show full details for result rank N");
    println!("  /help           - Show this help");
    println!("  Ctrl+D or Ctrl+C - Exit");
    println!();

    let mut rl = DefaultEditor::new()
        .with_context(|| "Failed to create readline editor")?;

    let mut last_results: Vec<SearchResult> = Vec::new();

    loop {
        let readline = rl.readline("Search> ");
        match readline {
            Ok(line) => {
                let line = line.trim();
                
                if line.is_empty() {
                    continue;
                }

                rl.add_history_entry(line)
                    .ok(); // Ignore errors from adding to history

                // Handle commands
                if line.starts_with('/') {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    match parts[0] {
                        "/help" => {
                            println!("Commands:");
                            println!("  <query>         - Search for papers");
                            println!("  /top N          - Set number of results to N");
                            println!("  /year START END - Filter by year range");
                            println!("  /year clear     - Clear year filter");
                            println!("  /format table   - Use table output format");
                            println!("  /format json    - Use JSON output format");
                            println!("  /detail N       - Show full details for result rank N");
                            println!("  /help           - Show this help");
                            println!("  Ctrl+D or Ctrl+C - Exit");
                        }
                        "/top" => {
                            if parts.len() != 2 {
                                eprintln!("Usage: /top N");
                                continue;
                            }
                            match parts[1].parse::<usize>() {
                                Ok(n) if n > 0 => {
                                    top_k = n;
                                    println!("Set top-k to {}", top_k);
                                }
                                _ => eprintln!("Invalid number: must be a positive integer"),
                            }
                        }
                        "/year" => {
                            if parts.len() == 2 && parts[1] == "clear" {
                                year_range = None;
                                println!("Cleared year filter");
                            } else if parts.len() == 3 {
                                match (parts[1].parse::<i32>(), parts[2].parse::<i32>()) {
                                    (Ok(start), Ok(end)) if start <= end => {
                                        year_range = Some(YearRange::new(start, end));
                                        println!("Set year filter: {} - {}", start, end);
                                    }
                                    _ => eprintln!("Invalid year range: START must be <= END"),
                                }
                            } else {
                                eprintln!("Usage: /year START END  or  /year clear");
                            }
                        }
                        "/format" => {
                            if parts.len() != 2 {
                                eprintln!("Usage: /format [table|json]");
                                continue;
                            }
                            match parts[1] {
                                "table" => {
                                    format = OutputFormat::Table;
                                    println!("Set output format to table");
                                }
                                "json" => {
                                    format = OutputFormat::Json;
                                    println!("Set output format to JSON");
                                }
                                _ => eprintln!("Invalid format: must be 'table' or 'json'"),
                            }
                        }
                        "/detail" => {
                            if parts.len() != 2 {
                                eprintln!("Usage: /detail N");
                                continue;
                            }
                            match parts[1].parse::<usize>() {
                                Ok(rank) if rank > 0 && rank <= last_results.len() => {
                                    display_result_detail(&last_results[rank - 1], rank);
                                }
                                Ok(rank) if rank > last_results.len() => {
                                    eprintln!("Rank {} out of range (last search had {} results)",
                                        rank, last_results.len());
                                }
                                _ => eprintln!("Invalid rank: must be a positive integer"),
                            }
                        }
                        _ => eprintln!("Unknown command: {}. Type /help for available commands.", parts[0]),
                    }
                } else {
                    // Execute search
                    let start = Instant::now();
                    match execute_search(&engine, line, top_k, year_range).await {
                        Ok(results) => {
                            let elapsed = start.elapsed();
                            last_results = results.clone();

                            match format {
                                OutputFormat::Table => {
                                    println!("{}", format_results_table(&results));
                                    println!(
                                        "\nFound {} results in {:.2}s",
                                        results.len(),
                                        elapsed.as_secs_f64()
                                    );
                                }
                                OutputFormat::Json => {
                                    match format_results_json(&results) {
                                        Ok(json) => println!("{}", json),
                                        Err(e) => eprintln!("Error formatting JSON: {}", e),
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Search failed: {}", e);
                        }
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                println!("Goodbye!");
                break;
            }
            Err(err) => {
                error!("Error reading input: {}", err);
                break;
            }
        }
    }

    Ok(())
}

/// Run single-query mode
async fn run_single_query<E: EmbeddingProvider, S: PaperStorage>(
    engine: BruteForceSearchEngine<E, S>,
    query: &str,
    top_k: usize,
    year_range: Option<YearRange>,
    format: OutputFormat,
) -> Result<()> {
    let start = Instant::now();
    let results = execute_search(&engine, query, top_k, year_range).await?;
    let elapsed = start.elapsed();

    match format {
        OutputFormat::Table => {
            println!("{}", format_results_table(&results));
            println!(
                "\nFound {} results in {:.2}s",
                results.len(),
                elapsed.as_secs_f64()
            );
        }
        OutputFormat::Json => {
            let json = format_results_json(&results)?;
            println!("{}", json);
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Setup logging
    setup_logging(&args.log_level);

    // Validate arguments
    if !args.interactive && args.query.is_none() {
        anyhow::bail!(
            "Either --query or --interactive must be specified.\n\
             Use --help for usage information."
        );
    }

    if let (Some(start), Some(end)) = (args.year_start, args.year_end) {
        if start > end {
            anyhow::bail!(
                "Invalid year range: start year ({}) cannot be greater than end year ({})",
                start,
                end
            );
        }
    }

    // Check database exists
    if !args.db_path.exists() {
        anyhow::bail!(
            "Database file not found: {}\n\
             Please run the ingestion binary first to create the database.",
            args.db_path.display()
        );
    }

    info!("Loading database from: {}", args.db_path.display());

    // Initialize storage
    let mut storage = SqliteStorage::new(args.db_path.to_string_lossy().to_string());
    storage
        .initialize()
        .await
        .with_context(|| "Failed to initialize storage")?;

    // Get embedding config from database
    let config = storage
        .get_config()
        .await
        .with_context(|| "Failed to retrieve embedding configuration from database")?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Database has no embedding configuration.\n\
                 The database may not have been properly initialized.\n\
                 Please run the ingestion binary with --mode init-new first."
            )
        })?;

    info!(
        "Using embedding model: {} (dimension: {})",
        config.model_name, config.dimension
    );

    // Validate database has papers
    let paper_count = storage
        .count_papers()
        .await
        .with_context(|| "Failed to count papers in database")?;

    if paper_count == 0 {
        anyhow::bail!(
            "Database is empty (0 papers found).\n\
             Please run the ingestion binary to add papers first."
        );
    }

    info!("Database contains {} papers", paper_count);

    // Auto-detect and create embedding provider
    let embedding_provider =
        create_embedding_provider(&config.model_name, config.dimension, args.cache_dir).await?;

    info!("Embedding provider initialized successfully");

    // Create search engine
    let engine = BruteForceSearchEngine::new(embedding_provider, storage);

    // Build year range if specified
    let year_range = match (args.year_start, args.year_end) {
        (Some(start), Some(end)) => Some(YearRange::new(start, end)),
        (Some(start), None) => Some(YearRange::new(start, i32::MAX)),
        (None, Some(end)) => Some(YearRange::new(i32::MIN, end)),
        (None, None) => None,
    };

    // Run in appropriate mode
    if args.interactive {
        run_interactive(engine, args.top_k, year_range, args.format).await?;
    } else {
        let query = args.query.unwrap(); // Safe because we validated above
        run_single_query(engine, &query, args.top_k, year_range, args.format).await?;
    }

    Ok(())
}
