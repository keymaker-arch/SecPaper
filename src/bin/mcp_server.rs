//! MCP server binary entry point.
//!
//! This binary starts the MCP server that exposes the paper search API.
//! It loads the pre-built database and handles incoming search requests.

use mcp_paper_search::server::{McpServer, ServerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO: Parse command-line arguments
    // TODO: Load configuration from file or environment
    // TODO: Set up logging
    
    println!("Starting MCP Paper Search Server...");
    
    // Create server with default configuration
    let config = ServerConfig::default();
    let mut server = McpServer::new(config);
    
    // Initialize and run
    server.initialize().await?;
    println!("Server initialized successfully");
    
    println!("Server listening on {}:{}", "127.0.0.1", 3000);
    server.run().await?;
    
    Ok(())
}
