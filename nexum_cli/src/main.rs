use nexum_core::{Catalog, Executor, NLTranslator, Parser, QueryExplainer, StorageEngine};
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    println!("NexumDB v0.2.0 - AI-Native Database with Natural Language Support");
    println!("Initializing...\n");

    let storage = StorageEngine::new("./nexumdb_data")?;
    let executor = Executor::new(storage.clone()).with_cache();
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

    let query_explainer = match QueryExplainer::new() {
        Ok(explainer) => {
            println!("Query EXPLAIN enabled");
            Some(explainer)
        }
        Err(e) => {
            println!("Warning: Query explainer not available: {}", e);
            None
        }
    };

    println!("Ready. Commands:");
    println!("  - SQL: Type any SQL query (CREATE TABLE, INSERT, SELECT)");
    println!("  - ASK: Type 'ASK <question>' for natural language queries");
    println!("  - EXPLAIN: Type 'EXPLAIN <query>' to see query execution plan");
    println!("  - EXIT: Type 'exit' or 'quit' to exit\n");

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

        // Handle EXPLAIN command
        if input.to_uppercase().starts_with("EXPLAIN ") {
            let query_to_explain = input[8..].trim();

            if let Some(ref explainer) = query_explainer {
                println!();
                match explainer.explain(query_to_explain) {
                    Ok(plan) => {
                        println!("{}", plan);
                    }
                    Err(e) => {
                        eprintln!("Explain error: {}", e);
                    }
                }
            } else {
                eprintln!("Query explainer not available");
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
