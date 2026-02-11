# MCP Paper Search Server - Refined Development Plan (Round 3)

## 1. Product Focus (MVP Definition)
Goal: Given a user query about a topic, return top-k relevant research papers based on abstract embeddings.

MVP scope:
- Single offline database, prebuilt before server start
- Search by abstract embeddings + optional keyword filters
- Basic metadata fields (title, authors, affiliations, abstract)
- Optional filter for publish year range

Out of scope for MVP:
- Full-text PDF parsing
- Citation graph or paper recommendation logic
- Multi-source ingestion (ArXiv API, Semantic Scholar, etc.)

The project will be implemented using Rust.

## 2. System Architecture
High-level design:
- Offline ingestion pipeline builds a deduplicated SQLite database with embeddings.
- MCP server loads the database and exposes a search API.
- A pluggable storage + vector search abstraction allows future backend upgrades.
- An embedding provider abstraction isolates text embedding computation from callers.

Data flow:
1. Input metadata -> normalization -> dedup -> embedding computation -> storage
2. Server starts -> loads DB connection and config -> ready to serve
3. Query -> normalization -> embedding -> vector search -> optional year filter -> top-k results

Workflow diagram (offline build and online query paths):
```
Offline build (before MCP server starts)
  +-----------------------+
  |  Paper Provider       |
  |  (JSON, API, etc.)    |
  +-----------+-----------+
              |
              v
      +-------------------+
      | Ingestion Pipeline|
      +---------+---------+
                |
                v
        +--------------+
        | Normalization|
        +------+-------+
               |
               v
         +-----------+
         |  Dedup    |
         +-----+-----+
               |
               v
     +-------------------+
     | Embedding Provider|
     +---------+---------+
               |
               v
       +---------------+
       |   SQLite DB   |
       +---------------+

Online query (after MCP server starts)
  +---------------+
  |   MCP Server  |
  +-------+-------+
          |
          v
  +---------------+
  |  User Query   |
  +-------+-------+
          |
          v
  +---------------+
  | Normalization |
  +-------+-------+
          |
          v
  +-------------------+
  | Embedding Provider|
  +---------+---------+
            |
            v
  +-------------------+
  | Vector Search     |
  | + Year Filter     |
  +---------+---------+
            |
            v
  +---------------+
  | Search Result |
  +---------------+
```

## 3. Module: Paper Provider
Design:
- Abstract interface (trait) for sourcing paper metadata from various providers.
- Each provider implementation handles its own data format and API interactions.
- Yields normalized paper metadata in a common format for ingestion.

Role:
- Decouples data sources from ingestion logic.
- Allows extensibility to support multiple paper databases (ArXiv, Semantic Scholar, etc.).
- Provides a uniform interface for the ingestion pipeline.

Tech details:
- Core trait: PaperProvider with methods to yield paper metadata.
- Paper metadata structure: title, authors (with affiliations), abstract, publish_year.
- Provider implementations handle pagination, rate limiting, and error recovery.
- MVP may include a simple JSON file provider for initial testing.
- Future providers: ArXiv API, Semantic Scholar API, PDF parsers, etc.

## 4. Module: Ingestion Pipeline
Design:
- Batch process input metadata into a normalized, deduplicated dataset.
- Uses the Paper Provider abstraction to fetch papers from various sources.
- Compute and store embeddings alongside metadata.
- Primary use case: Connect to existing storage and add new papers while respecting the stored embedding configuration.
- Secondary use case: Initialize new storage with a specified embedding provider.

Role:
- Adds papers to the searchable database.
- Enforces deduplication to avoid re-ingesting the same paper.
- Integrates with any PaperProvider implementation to support multiple data sources.
- Validates embedding provider consistency with stored configuration.

Tech details:
- Deduplication key: normalized paper title (lowercase, trimmed, collapsed spaces).
- Embedding configuration stored in database (model name + vector dimension).
- Embeddings stored as float32 arrays serialized into BLOB.
- Primary workflow: `IngestionPipeline::connect()` -> validates embedding config -> `ingest_from_provider()`.
- Secondary workflow: `IngestionPipeline::initialize_new()` -> stores embedding config -> `ingest_from_provider()`.
- Statistics tracking: counts inserted, duplicates, and failed papers.
- The pipeline reads embedding config from storage and validates that the provided embedding provider matches it.
- For new storage, the pipeline initializes schema and stores the embedding provider's configuration.

### Ingestion CLI Binary
The `ingestion` binary provides a production-ready command-line interface for building and updating the paper database.

**Command-Line Interface:**
- `--input <FILE>`: Input JSON file containing paper metadata (required)
- `--db-path <PATH>`: Database file path (default: `papers.db`)
- `--mode <MODE>`: Operation mode - `connect` (default) or `init-new`
- `--embedding-provider <PROVIDER>`: Embedding provider - `fastembed` (default) or `openai`
- `--embedding-model <MODEL>`: Specific model name (optional, provider-dependent)
- `--batch-size <N>`: Papers per embedding batch (default: 100)
- `--log-level <LEVEL>`: Logging verbosity - error, warn, info, debug, trace (default: info)
- `--cache-dir <DIR>`: FastEmbed model cache directory (default: `~/.cache/fastembed`)

**Operation Modes:**
- **connect** (default): Connect to existing database, validate embedding config matches, add new papers
- **init-new**: Initialize new database with fresh schema and embedding configuration

**Features:**
- Automatic embedding config validation prevents mixing different embedding models
- Real-time progress tracking with detailed statistics
- Structured logging with adjustable verbosity levels
- Automatic parent directory creation for database path
- Comprehensive error messages with actionable suggestions
- Deduplication in connect mode prevents re-ingesting existing papers

**Default Configuration:**
- Embedding provider: FastEmbed (AllMiniLML6V2, 384 dimensions)
- Storage: SQLite
- Paper provider: JSON file
- Batch size: 100 papers
- Log level: info

**Example Commands:**
```bash
# Initialize new database with FastEmbed
ingestion --mode init-new --input papers.json --db-path research.db

# Add papers to existing database
ingestion --input new_papers.json --db-path research.db

# Use OpenAI embeddings with custom batch size
OPENAI_API_KEY=sk-... ingestion --mode init-new --input papers.json \
  --embedding-provider openai --batch-size 50

# Debug mode with trace logging
ingestion --input papers.json --log-level trace
```

## 5. Module: Storage Layer
Design:
- SQLite v1 schema optimized for MVP requirements.
- A storage interface abstracts future backends (e.g., vector DBs).

Role:
- Persists metadata and embeddings.
- Supports scan + similarity ranking for MVP.

Tech details:
Table: papers
- id (INTEGER PRIMARY KEY)
- title (TEXT)
- authors (TEXT)  -- JSON encoded list, including author name and affiliations
- abstract (TEXT)
- publish_year (INTEGER)  -- used for optional range filter
- embedding (BLOB)  -- serialized float32

## 6. Module: Embedding Strategy
Design:
- Embedding computation isolated behind a trait/interface.
- Text normalization shared between ingestion and query paths.
- Embedding provider abstraction so callers depend on a stable API, not a specific model.
- Multiple provider implementations available:
  - OpenAI: Cloud-based API using OpenAI's text-embedding models
  - FastEmbed: Local embedding generation using the fastembed Rust library

Role:
- Produces consistent vectors for papers and user queries.
- Allows future swap to local or hosted models.
- Provides flexibility between API-based and local embedding generation.

Tech details:
- OpenAI Model: text-embedding-3-small (1536 dimensions).
- FastEmbed Default Model: AllMiniLML6V2 (384 dimensions).
- Normalization: lowercase, trim, collapse spaces.
- Store model name and vector dimension in config.
- FastEmbed advantages:
  - No API costs
  - Runs locally without internet connection
  - Better for batch processing large datasets
  - Supports multiple open-source embedding models

## 7. Module: Query & Ranking

**Status:** ✅ Implemented

Design:
- Compute query embedding and rank by cosine similarity.
- Apply optional publish year range filter.
- Return top-k results with scores and relevance labels.

Role:
- Provides core search functionality for the MCP server.

Implementation Details:
- `BruteForceSearchEngine` implementation coordinates `EmbeddingProvider` and `PaperStorage`
- Text normalization using `normalize_text()` applied to queries before embedding
- Brute-force similarity computation across all papers (optimized via storage-level year filtering)
- Results sorted by descending cosine similarity score
- `RelevanceLevel` automatically assigned via score thresholds (IDENTICAL >0.95, HIGHLY_SIMILAR >0.85, SIMILAR >0.70, RELEVANT ≥0.0)
- Error handling converts embedding and storage errors to `QueryError` variants
- Comprehensive test coverage including mocks for embedding and storage providers

Tech details:
- MVP uses brute-force similarity search in SQLite.
- Keep vector search interface to swap in Qdrant or similar later.
- Relevance labels: IDENTICAL, HIGHLY_SIMILAR, SIMILAR, RELEVANT (expandable).
- Cosine similarity function tested and validated with unit tests.
- Full documentation with usage examples in module header.

## 8. Module: MCP Server API
Design:
- Single MCP endpoint for search, with future room for admin endpoints.

Role:
- Receives user queries and returns ranked papers.

Tech details:
Core endpoint: search_papers
Input:
- query (string)
- paper_count (int, default 10)
- publish_year_range (optional, [start_year, end_year])
Output:
- list of {title, authors, abstract, relevance, score}

## Project Structure

```
SecPaper/
├── Cargo.toml                  # Project manifest with dependencies
├── README.md                   # Project documentation
├── LICENSE-MIT                 # MIT license
├── LICENSE-APACHE              # Apache 2.0 license
├── .gitignore                  # Git ignore patterns
├── PLAN.md                     # Original development plan
│
└── src/
    ├── lib.rs                  # Library root with module declarations
    │
    ├── models/
    │   └── mod.rs              # Core data structures (Paper, Author, SearchResult, etc.)
    │
    ├── provider/
    │   ├── mod.rs              # PaperProvider trait and error types
    │   └── json.rs             # JSON file provider implementation
    │
    ├── embedding/
    │   ├── mod.rs              # EmbeddingProvider trait and text normalization
    │   ├── openai.rs           # OpenAI API implementation
    │   └── fastembed.rs        # FastEmbed local embedding implementation
    │
    ├── storage/
    │   ├── mod.rs              # PaperStorage trait and types
    │   └── sqlite.rs           # SQLite storage implementation
    │
    ├── query/
    │   └── mod.rs              # ✅ SearchEngine trait and BruteForceSearchEngine (IMPLEMENTED)
    │
    ├── ingestion/
    │   └── mod.rs              # IngestionPipeline for offline processing
    │
    ├── server/
    │   └── mod.rs              # MCP server types and request/response DTOs
    │
    └── bin/
        ├── mcp_server.rs       # MCP server binary entry point
        └── ingestion.rs        # Ingestion pipeline binary entry point
```

## Module Overview

### Core Modules

- **models**: Defines all data structures used across the application
  - `Paper`: Paper metadata with embeddings
  - `Author`: Author name and affiliation
  - `SearchResult`: Search result with relevance scoring
  - `RelevanceLevel`: Categorical relevance classification
  - `EmbeddingConfig`: Embedding model configuration

- **provider**: Paper metadata source abstraction
  - `PaperProvider` trait: Interface for sourcing papers from various backends
  - `ProviderError`: Error types for provider operations
  - `JsonFileProvider`: JSON file-based provider for testing and offline datasets
  - Future providers: ArXiv API, Semantic Scholar API, PDF parsers

- **embedding**: Text embedding abstraction
  - `EmbeddingProvider` trait: Interface for embedding generation
  - `normalize_text()`: Text normalization utility
  - `OpenAIEmbedding`: OpenAI API implementation
  - `FastEmbedProvider`: Local embedding generation using fastembed library

- **storage**: Data persistence layer
  - `PaperStorage` trait: Interface for paper storage
  - `YearRange`: Publication year filter
  - `SqliteStorage`: SQLite backend implementation

- **query**: Search and ranking
  - `SearchEngine` trait: Interface for search execution
  - `SearchQuery`: Query parameters
  - `BruteForceSearchEngine`: MVP implementation
  - `cosine_similarity()`: Similarity computation

- **ingestion**: Offline data processing
  - `IngestionPipeline`: Coordinates embedding and storage
  - `IngestionStats`: Processing statistics
  - `connect()`: Primary method to connect to existing storage and validate embedding config
  - `initialize_new()`: Secondary method to initialize new storage with embedding config
  - `ingest_from_provider()`: Main ingestion method that fetches papers from a PaperProvider
  - `ingest_batch()`: Batch processing method for direct paper processing
  - Deduplication by normalized title
  - Validates embedding provider consistency with stored configuration

- **server**: MCP server implementation
  - `McpServer`: Server instance
  - `SearchPapersRequest/Response`: API request/response types
  - `ServerConfig`: Server configuration

### Binary Targets

- **mcp_server**: Starts the MCP server for handling search queries
  - Entry point: `src/bin/mcp_server.rs`
  - Loads database and exposes search API
  - Configurable host, port, and concurrency settings

- **ingestion**: CLI tool for building and updating the paper database
  - Entry point: `src/bin/ingestion.rs`
  - Supports two modes: `connect` (add to existing) and `init-new` (create database)
  - Configurable embedding provider (FastEmbed default, OpenAI optional)
  - Progress tracking with detailed statistics display
  - Validates embedding config consistency when adding to existing database
  - Structured logging with adjustable verbosity (error/warn/info/debug/trace)
  - Built with `clap` for robust CLI argument parsing
  - Example: `ingestion --input papers.json --db-path research.db`
