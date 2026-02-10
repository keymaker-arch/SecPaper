# MCP Paper Search Server

A semantic search engine for research papers using abstract embeddings, implemented as an MCP (Model Context Protocol) server.

## Overview

This project provides semantic search over research papers by:
- Generating vector embeddings from paper abstracts
- Storing papers and embeddings in a SQLite database
- Exposing a search API via MCP protocol
- Ranking results by cosine similarity

## Architecture

The system consists of two main components:

1. **Ingestion Pipeline** (`ingestion` binary): Offline process that builds the searchable database
2. **MCP Server** (`mcp-server` binary): Online service that handles search queries

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
- OpenAI API key (for embedding generation)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SecPaper
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key-here
```

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

Run the ingestion pipeline to build the database:

```bash
cargo run --bin ingestion -- --input papers.json --output papers.db
```

Input format should be a JSON array of papers:
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

This is the initial skeleton/interface definition. Core implementations are marked with `TODO` and `unimplemented!()`.

## License

MIT OR Apache-2.0
