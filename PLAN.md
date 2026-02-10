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
1. Input metadata -> normalization -> dedup -> embedding computation -> SQLite storage
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
- Persist all records into a single SQLite file before server start.

Role:
- Produces the searchable offline database.
- Enforces deduplication to avoid re-ingesting the same paper.
- Integrates with any PaperProvider implementation to support multiple data sources.

Tech details:
- Deduplication key: normalized paper title (lowercase, trimmed, collapsed spaces).
- Embedding model: OpenAI text-embedding-3-small.
- Embeddings stored as float32 arrays serialized into BLOB.
- Outputs one SQLite file and a small config file with model name + vector dimension.
- Primary method: `ingest_from_provider()` accepts a PaperProvider reference and processes all papers.
- Legacy method: `ingest_batch()` accepts a slice of papers directly (for advanced use cases).
- Statistics tracking: counts inserted, duplicates, and failed papers.

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

Role:
- Produces consistent vectors for papers and user queries.
- Allows future swap to local or hosted models.

Tech details:
- Model: OpenAI text-embedding-3-small.
- Normalization: lowercase, trim, collapse spaces.
- Store model name and vector dimension in config.

## 7. Module: Query & Ranking
Design:
- Compute query embedding and rank by cosine similarity.
- Apply optional publish year range filter.
- Return top-k results with scores and relevance labels.

Role:
- Provides core search functionality for the MCP server.

Tech details:
- MVP uses brute-force similarity search in SQLite.
- Keep vector search interface to swap in Qdrant or similar later.
- Relevance labels: IDENTICAL, HIGHLY_SIMILAR, SIMILAR, RELEVANT (expandable).

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
