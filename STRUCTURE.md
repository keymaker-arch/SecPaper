# Project Structure

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
    │   └── openai.rs           # OpenAI API implementation
    │
    ├── storage/
    │   ├── mod.rs              # PaperStorage trait and types
    │   └── sqlite.rs           # SQLite storage implementation
    │
    ├── query/
    │   └── mod.rs              # SearchEngine trait and BruteForceSearchEngine
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
  - `ingest_from_provider()`: Primary ingestion method that fetches papers from a PaperProvider
  - `ingest_batch()`: Legacy method for direct paper processing
  - Deduplication by normalized title

- **server**: MCP server implementation
  - `McpServer`: Server instance
  - `SearchPapersRequest/Response`: API request/response types
  - `ServerConfig`: Server configuration

### Binary Targets

- **mcp_server**: Starts the MCP server for handling search queries
- **ingestion**: Runs the offline pipeline to build the database

## Key Design Patterns

1. **Trait-based abstraction**: Storage, embedding, provider, and search are all behind traits
2. **Async-first**: All I/O operations are async using tokio
3. **Type safety**: Strong typing with serde for serialization
4. **Error handling**: Custom error types with thiserror
5. **Separation of concerns**: Clear module boundaries
6. **Provider pattern**: Ingestion pipeline delegates paper sourcing to PaperProvider implementations

## Implementation Status

All interfaces and data structures are defined with:
- Complete type signatures
- Comprehensive documentation
- Unimplemented function bodies (marked with `TODO`)
- Unit tests for pure functions

Ready for incremental implementation of each module.
