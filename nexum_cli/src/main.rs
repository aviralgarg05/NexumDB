use nexum_core::{Catalog, Executor, NLTranslator, Parser, StorageEngine};
use std::env;
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    println!("NexumDB v0.2.0 - AI-Native Database with Natural Language Support");
    println!("Initializing...\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let no_cache = args.iter().any(|arg| arg == "--no-cache");

    if no_cache {
        println!("Result caching disabled (--no-cache flag)");
    }

    let storage = StorageEngine::new("./nexumdb_data")?;

    let executor = if no_cache {
        Executor::new_with_cache_disabled(storage.clone())
    } else {
        Executor::new(storage.clone()).with_cache()
    };

    let catalog = Catalog::new(storage);

    let nl_translator = match NLTranslator::new() {
        Ok(translator) => {
            println!("Natural Language translator enabled");
            Some(translator)
        }
        Err(e) => {
            println!("Warning: NL translator not available: {}", e);
            None
        }
    };

    println!("Ready. Commands:");
    println!("  - SQL: Type any SQL query (CREATE TABLE, INSERT, SELECT)");
    println!("  - ASK: Type 'ASK <question>' for natural language queries");
    println!("  - CACHE: Type 'CACHE STATS' or 'CACHE CLEAR' to manage cache");
    println!("  - EXIT: Type 'exit' or 'quit' to exit");
    if !no_cache {
        println!("  - Use --no-cache flag to disable result caching");
    }
    println!();

    loop {
        print!("nexumdb> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }

        // Handle cache commands
        if input.to_uppercase().starts_with("CACHE ") {
            let cache_cmd = input[6..].trim().to_uppercase();

            match cache_cmd.as_str() {
                "STATS" => match executor.cache_stats() {
                    Ok(stats) => {
                        println!("Cache Statistics:");
                        println!("  Total entries: {}", stats.total_entries);
                        println!("  Total size: {} bytes", stats.total_size_bytes);
                        if let Some(oldest) = stats.oldest_entry_timestamp {
                            println!("  Oldest entry: {} (unix timestamp)", oldest);
                        }
                        if let Some(newest) = stats.newest_entry_timestamp {
                            println!("  Newest entry: {} (unix timestamp)", newest);
                        }
                    }
                    Err(e) => eprintln!("Error getting cache stats: {}", e),
                },
                "CLEAR" => match executor.clear_cache() {
                    Ok(_) => println!("Cache cleared successfully"),
                    Err(e) => eprintln!("Error clearing cache: {}", e),
                },
                _ => {
                    eprintln!("Unknown cache command. Use 'CACHE STATS' or 'CACHE CLEAR'");
                }
            }
            continue;
        }

        if input.to_uppercase().starts_with("ASK ") {
            let natural_query = input[4..].trim();

            if let Some(ref translator) = nl_translator {
                let schema = get_schema_context(&catalog);

                println!("Translating: '{}'", natural_query);
                match translator.translate(natural_query, &schema) {
                    Ok(sql) => {
                        println!("Generated SQL: {}", sql);
                        println!();

                        match Parser::parse(&sql) {
                            Ok(statement) => match executor.execute(statement) {
                                Ok(result) => {
                                    println!("{:?}", result);
                                }
                                Err(e) => {
                                    eprintln!("Execution error: {}", e);
                                }
                            },
                            Err(e) => {
                                eprintln!("Parse error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Translation error: {}", e);
                    }
                }
            } else {
                eprintln!("Natural language translator not available");
            }
            continue;
        }

        match Parser::parse(input) {
            Ok(statement) => match executor.execute(statement) {
                Ok(result) => {
                    println!("{:?}", result);
                }
                Err(e) => {
                    eprintln!("Execution error: {}", e);
                }
            },
            Err(e) => {
                eprintln!("Parse error: {}", e);
            }
        }
    }

    Ok(())
}

fn get_schema_context(catalog: &Catalog) -> String {
    match catalog.list_tables() {
        Ok(tables) => {
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
                    schema.push_str(")\n");
                }
            }
            schema
        }
        Err(_) => String::new(),
    }
}
