use nexum_core::{Catalog, Executor, NLTranslator, Parser, QueryExplainer, StorageEngine};
use std::io::{self, Write};
use tracing::{debug, error, info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

fn init_logging() {
    // Try to read NEXUM_LOG_FORMAT environment variable (json or pretty)
    let log_format = std::env::var("NEXUM_LOG_FORMAT").unwrap_or_else(|_| "pretty".to_string());
    
    // Create env filter with default level INFO, configurable via RUST_LOG
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    match log_format.to_lowercase().as_str() {
        "json" => {
            // Structured JSON logging for production/monitoring
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt::layer().json())
                .init();
        }
        _ => {
            // Pretty formatted logging for development
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt::layer().pretty())
                .init();
        }
    }
}

fn main() -> anyhow::Result<()> {
    // Initialize logging system
    init_logging();
    
    info!("Starting NexumDB v0.2.0 - AI-Native Database with Natural Language Support");
    
    println!("NexumDB v0.2.0 - AI-Native Database with Natural Language Support");
    println!("Initializing...\\n");

    debug!("Initializing storage engine at ./nexumdb_data");
    let storage = StorageEngine::new("./nexumdb_data")?;
    info!("Storage engine initialized successfully");
    
    debug!("Creating executor with cache enabled");
    let executor = Executor::new(storage.clone()).with_cache();
    let catalog = Catalog::new(storage);
    info!("Executor and catalog initialized");

    let nl_translator = match NLTranslator::new() {
        Ok(translator) => {
            info!("Natural Language translator enabled");
            println!("Natural Language translator enabled");
            Some(translator)
        }
        Err(e) => {
            warn!("NL translator not available: {}", e);
            println!("Warning: NL translator not available: {}", e);
            None
        }
    };

    let query_explainer = match QueryExplainer::new() {
        Ok(explainer) => {
            info!("Query EXPLAIN feature enabled");
            println!("Query EXPLAIN enabled");
            Some(explainer)
        }
        Err(e) => {
            warn!("Query explainer not available: {}", e);
            println!("Warning: Query explainer not available: {}", e);
            None
        }
    };

    println!("Ready. Commands:");
    println!("  - SQL: Type any SQL query (CREATE TABLE, INSERT, SELECT)");
    println!("  - ASK: Type 'ASK <question>' for natural language queries");
    println!("  - EXPLAIN: Type 'EXPLAIN <query>' to see query execution plan");
    println!("  - EXIT: Type 'exit' or 'quit' to exit\\n");
    
    info!("REPL ready, waiting for user input");

    loop {
        print!("nexumdb> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        debug!("Received input: {}", input);

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            info!("User requested exit");
            println!("Goodbye!");
            break;
        }

        if input.to_uppercase().starts_with("ASK ") {
            let natural_query = input[4..].trim();
            info!("Processing natural language query: {}", natural_query);

            if let Some(ref translator) = nl_translator {
                let schema = get_schema_context(&catalog);
                debug!("Schema context: {}", schema);

                println!("Translating: '{}'", natural_query);
                match translator.translate(natural_query, &schema) {
                    Ok(sql) => {
                        info!("Translation successful: {}", sql);
                        println!("Generated SQL: {}", sql);
                        println!();

                        match Parser::parse(&sql) {
                            Ok(statement) => {
                                debug!("SQL parsed successfully");
                                match executor.execute(statement) {
                                    Ok(result) => {
                                        info!("Query executed successfully");
                                        println!("{:?}", result);
                                    }
                                    Err(e) => {
                                        error!("Execution error: {}", e);
                                        eprintln!("Execution error: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Parse error: {}", e);
                                eprintln!("Parse error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Translation error: {}", e);
                        eprintln!("Translation error: {}", e);
                    }
                }
            } else {
                warn!("Natural language translator not available for query");
                eprintln!("Natural language translator not available");
            }
            continue;
        }

        // Handle EXPLAIN command
        if input.to_uppercase().starts_with("EXPLAIN ") {
            let query_to_explain = input[8..].trim();
            info!("Processing EXPLAIN command for: {}", query_to_explain);

            if let Some(ref explainer) = query_explainer {
                println!();
                match explainer.explain(query_to_explain) {
                    Ok(plan) => {
                        info!("Query plan generated successfully");
                        println!("{}", plan);
                    }
                    Err(e) => {
                        error!("Explain error: {}", e);
                        eprintln!("Explain error: {}", e);
                    }
                }
            } else {
                warn!("Query explainer not available");
                eprintln!("Query explainer not available");
            }
            continue;
        }

        // Regular SQL query
        info!("Processing SQL query: {}", input);
        match Parser::parse(input) {
            Ok(statement) => {
                debug!("SQL parsed successfully: {:?}", statement);
                match executor.execute(statement) {
                    Ok(result) => {
                        info!("Query executed successfully");
                        println!("{:?}", result);
                    }
                    Err(e) => {
                        error!("Execution error: {}", e);
                        eprintln!("Execution error: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Parse error: {}", e);
                eprintln!("Parse error: {}", e);
            }
        }
    }

    info!("NexumDB shutting down");
    Ok(())
}

fn get_schema_context(catalog: &Catalog) -> String {
    debug!("Retrieving schema context");
    match catalog.list_tables() {
        Ok(tables) => {
            debug!("Found {} tables", tables.len());
            let mut schema = String::new();
            for table_name in tables {
                if let Ok(Some(table_schema)) = catalog.get_table(&table_name) {
                    schema.push_str(&format!("TABLE {} (", table_schema.name));
                    let columns: Vec<String> = table_schema
                        .columns
                        .iter()
                        .map(|c| format!("{} {:?}", c.name, c.data_type))
                        .collect();
                    schema.push_str(&columns.join(", "));
                    schema.push_str(")\\n");
                }
            }
            debug!("Schema context generated: {} bytes", schema.len());
            schema
        }
        Err(e) => {
            warn!("Failed to retrieve schema context: {}", e);
            String::new()
        }
    }
}
