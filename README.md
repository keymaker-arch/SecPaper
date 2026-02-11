# MCP Paper Search Server

A semantic search engine for research papers using abstract embeddings, implemented as an MCP (Model Context Protocol) server.

## Overview

This project provides semantic search over research papers by:
- Generating vector embeddings from paper abstracts
- Storing papers and embeddings in a SQLite database
- Exposing a search API via MCP protocol
- Ranking results by cosine similarity

## Architecture

The system consists of three main components:

1. **Ingestion Pipeline** (`ingestion` binary): Offline process that builds the searchable database
   - Uses PaperProvider abstraction to fetch papers from various sources (JSON files, APIs, etc.)
   - Normalizes and deduplicates papers
   - Generates embeddings via EmbeddingProvider (FastEmbed or OpenAI)
   - Persists to SQLite storage

2. **Search Tool** (`search` binary): Command-line interface for querying the database
   - Auto-detects embedding provider from database configuration
   - Supports both single-query and interactive REPL modes
   - Provides table or JSON output formats
   - Filters by publication year range

3. **MCP Server** (`mcp-server` binary): Online service that handles search queries via MCP protocol (planned)

### Modules

- `models`: Core data structures (Paper, Author, SearchResult)
- `provider`: Paper metadata sources (JSON files, APIs, etc.)
- `embedding`: Text embedding generation (OpenAI integration)
- `storage`: SQLite-based persistence layer
- `query`: Search execution and ranking
- `ingestion`: Offline data processing pipeline
- `server`: MCP server implementation

## Prerequisites

- Rust 1.70 or later
- OpenAI API key (optional, only if using OpenAI embeddings)

The default embedding provider is FastEmbed, which runs locally without requiring any API keys.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SecPaper
```

2. (Optional) Set your OpenAI API key if using OpenAI embeddings:
```bash
export OPENAI_API_KEY=your-api-key-here
```

Note: The default embedding provider (FastEmbed) runs locally and doesn't require an API key.

## Building

Build the project:
```bash
cargo build --release
```

## Usage

### Data Preparation

Prepare your paper metadata in JSON format. Create a file (e.g., `papers.json`) with an array of papers:

```json
[
  {
    "title": "Paper Title",
    "authors": [
      {"name": "Author Name", "affiliation": "University"}
    ],
    "abstract_text": "Paper abstract...",
    "publish_year": 2023
  }
]
```

See [papers.example.json](papers.example.json) for a complete example with real papers.

### Ingestion Pipeline

Initialize a new database with papers:

```bash
# Using FastEmbed (default, no API key needed)
cargo run --bin ingestion -- --mode init-new --input papers.json --db-path papers.db

# Using OpenAI embeddings (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your-api-key-here
cargo run --bin ingestion -- --mode init-new --input papers.json --db-path papers.db --embedding-provider openai
```

Add more papers to an existing database:

```bash
cargo run --bin ingestion -- --input new_papers.json --db-path papers.db
```

The ingestion pipeline:
1. Loads papers from the specified input source using a PaperProvider (currently JSON files)
2. Normalizes paper titles and checks for duplicates
3. Generates embeddings for paper abstracts (FastEmbed by default, or OpenAI)
4. Stores papers and embeddings in a SQLite database

### Searching Papers

Once you have built the database, you can search it using the `search` binary.

#### Single-Query Mode

Execute a single search and exit:

```bash
# Basic search
cargo run --bin search -- --db-path papers.db --query "neural networks"

# Search with more results
cargo run --bin search -- --db-path papers.db --query "transformers" --top-k 20

# Filter by publication year
cargo run --bin search -- --db-path papers.db --query "deep learning" --year-start 2020

# JSON output for scripting
cargo run --bin search -- --db-path papers.db --query "NLP" --format json
```

#### Interactive Mode

Start an interactive session for multiple queries:

```bash
cargo run --bin search -- --db-path papers.db --interactive
```

In interactive mode, you can:
- Type a query and press Enter to search
- `/top N` - Change the number of results
- `/year START END` - Set year filter
- `/year clear` - Clear year filter
- `/format table` or `/format json` - Toggle output format
- `/detail N` - Show full details for result rank N
- `/help` - Show available commands
- Ctrl+D or Ctrl+C - Exit

Example interactive session:
```
Search> transformers in NLP
[Table with top 10 results displayed]

Search> /top 5
Set top-k to 5

Search> /year 2020 2024
Set year filter: 2020 - 2024

Search> attention mechanisms
[Table with top 5 results from 2020-2024 displayed]

Search> /detail 1
[Full details of first result including complete abstract]
```

### MCP Server

Start the MCP server:

```bash
cargo run --bin mcp-server -- --database papers.db
```

### Search Query Example

Send a search request to the MCP server:

```json
{
  "query": "deep learning for natural language processing",
  "paper_count": 10,
  "publish_year_range": [2020, 2023]
}
```

## Configuration

Configuration options can be set via:
- Command-line arguments
- Environment variables
- Configuration file (TODO)

## Development

Run tests:
```bash
cargo test
```

Run with logging:
```bash
RUST_LOG=debug cargo run --bin mcp-server
```

## Project Status

**Implemented:**
- ✅ Complete data models and error types
- ✅ Storage layer with SQLite backend
- ✅ Embedding providers (FastEmbed and OpenAI)
- ✅ Paper provider abstraction with JSON file implementation
- ✅ Ingestion pipeline with deduplication and validation
- ✅ Query and ranking engine (BruteForceSearchEngine)
- ✅ Search binary with interactive and single-query modes

**In Progress:**
- 🚧 MCP server implementation

**Planned:**
- 📋 Additional paper providers (ArXiv API, Semantic Scholar)
- 📋 Vector database backends (Qdrant, Milvus)
- 📋 Citation graph features

## License

MIT OR Apache-2.0
