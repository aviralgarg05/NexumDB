use clap::Parser;
use colored::*;
use comfy_table::{Cell, Color, ContentArrangement, Table};
use indicatif::{ProgressBar, ProgressStyle};
use nexum_core::{
    executor::ExecutionResult, Catalog, Executor, NLTranslator, Parser as SqlParser,
    QueryExplainer, StorageEngine,
};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper};
use std::rc::Rc;
use std::time::Duration;

/// NexumDB - AI-Native Database with Natural Language Support
#[derive(Parser, Debug)]
#[command(name = "nexum")]
#[command(author = "Aviral Garg")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "An AI-native database combining SQL with semantic caching and NL queries")]
#[command(long_about = r#"
NexumDB is an innovative, open-source database that combines traditional SQL 
with AI-powered features including:

  • Semantic query caching using local embedding models
  • Natural language to SQL translation
  • Reinforcement learning-based query optimization
  • Advanced SQL operators (LIKE, IN, BETWEEN, ORDER BY, LIMIT)

EXAMPLES:
  nexum                  Start interactive REPL
  nexum --json           Output results as JSON
  nexum --help           Show this help message

SQL COMMANDS:
  CREATE TABLE name (col TYPE, ...)    Create a new table
  INSERT INTO name VALUES (...)        Insert rows
  SELECT * FROM name WHERE ...         Query data
  UPDATE name SET col = val WHERE ...  Update rows
  DELETE FROM name WHERE ...           Delete rows
  DROP TABLE name                      Delete a table

SPECIAL COMMANDS:
  SHOW TABLES            List all tables
  DESCRIBE <table>       Show table schema
  BEGIN / COMMIT / ROLLBACK  Transaction control
  ASK <question>         Natural language query
  EXPLAIN <query>        Show query execution plan
"#)]
struct Args {
    /// Output results in JSON format
    #[arg(short, long, default_value_t = false)]
    json: bool,
}

/// SQL keywords for auto-completion.
const SQL_KEYWORDS: &[&str] = &[
    "SELECT",
    "INSERT",
    "INTO",
    "VALUES",
    "CREATE",
    "TABLE",
    "DROP",
    "DELETE",
    "FROM",
    "WHERE",
    "UPDATE",
    "SET",
    "AND",
    "OR",
    "NOT",
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "LIMIT",
    "LIKE",
    "IN",
    "BETWEEN",
    "JOIN",
    "ON",
    "GROUP",
    "HAVING",
    "DISTINCT",
    "AS",
    "NULL",
    "INTEGER",
    "FLOAT",
    "TEXT",
    "BOOLEAN",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SHOW",
    "TABLES",
    "DESCRIBE",
    "IF",
    "EXISTS",
    "EXPLAIN",
    "ASK",
    "TRANSACTION",
];

/// Rustyline helper that provides tab-completion for SQL keywords,
/// table names from the catalog, and column names when a table context
/// can be inferred.
struct NexumCompleter {
    catalog: Rc<Catalog>,
}

impl NexumCompleter {
    fn new(catalog: Rc<Catalog>) -> Self {
        Self { catalog }
    }

    /// Try to detect a table name from the input so we can offer column completions.
    /// Looks for patterns like `FROM <table>`, `INTO <table>`, `UPDATE <table>`,
    /// `TABLE <table>`, `DESCRIBE <table>`.
    fn detect_table_context(&self, line: &str) -> Option<String> {
        let upper = line.to_uppercase();
        let context_keywords = ["FROM", "INTO", "UPDATE", "TABLE", "DESCRIBE"];
        for kw in &context_keywords {
            if let Some(pos) = upper.rfind(kw) {
                let after = &line[pos + kw.len()..];
                let table_name = after.split_whitespace().next();
                if let Some(name) = table_name {
                    if !name.is_empty() {
                        return Some(name.to_string());
                    }
                }
            }
        }
        None
    }
}

impl Completer for NexumCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let line_up_to_cursor = &line[..pos];
        // Find the start of the current word being typed
        let word_start = line_up_to_cursor
            .rfind(|c: char| c.is_whitespace() || c == ',' || c == '(')
            .map(|i| i + 1)
            .unwrap_or(0);
        let prefix = &line_up_to_cursor[word_start..];

        if prefix.is_empty() {
            return Ok((pos, vec![]));
        }

        let prefix_upper = prefix.to_uppercase();
        let mut candidates: Vec<Pair> = Vec::new();

        // 1. SQL keyword completions
        for kw in SQL_KEYWORDS {
            if kw.starts_with(&prefix_upper) {
                candidates.push(Pair {
                    display: kw.to_string(),
                    replacement: kw.to_string(),
                });
            }
        }

        // 2. Table name completions
        if let Ok(tables) = self.catalog.list_tables() {
            for table in &tables {
                let table_upper = table.to_uppercase();
                if table_upper.starts_with(&prefix_upper) {
                    candidates.push(Pair {
                        display: table.clone(),
                        replacement: table.clone(),
                    });
                }
            }
        }

        // 3. Column name completions when a table context is available
        if let Some(table_name) = self.detect_table_context(line_up_to_cursor) {
            if let Ok(Some(schema)) = self.catalog.get_table(&table_name) {
                for col in &schema.columns {
                    let col_upper = col.name.to_uppercase();
                    if col_upper.starts_with(&prefix_upper) {
                        candidates.push(Pair {
                            display: col.name.clone(),
                            replacement: col.name.clone(),
                        });
                    }
                }
            }
        }

        Ok((word_start, candidates))
    }
}

impl Hinter for NexumCompleter {
    type Hint = String;
}

impl Highlighter for NexumCompleter {}
impl Validator for NexumCompleter {}
impl Helper for NexumCompleter {}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    print_banner();

    let spinner = create_spinner("Initializing database engine...");
    let storage = StorageEngine::new("./nexumdb_data")?;
    let executor = Executor::new(storage.clone()).with_cache();
    let catalog = Rc::new(Catalog::new(storage));
    spinner.finish_with_message("✓ Database engine ready".green().to_string());

    let spinner = create_spinner("Loading NL translator...");
    let nl_translator = match NLTranslator::new() {
        Ok(translator) => {
            spinner
                .finish_with_message("✓ Natural Language translator enabled".green().to_string());
            Some(translator)
        }
        Err(e) => {
            spinner.finish_with_message(
                format!("⚠ NL translator not available: {}", e)
                    .yellow()
                    .to_string(),
            );
            None
        }
    };

    let spinner = create_spinner("Loading query explainer...");
    let query_explainer = match QueryExplainer::new() {
        Ok(explainer) => {
            spinner.finish_with_message("✓ Query EXPLAIN enabled".green().to_string());
            Some(explainer)
        }
        Err(e) => {
            spinner.finish_with_message(
                format!("⚠ Query explainer not available: {}", e)
                    .yellow()
                    .to_string(),
            );
            None
        }
    };

    // Set up rustyline editor with auto-completion
    let completer = NexumCompleter::new(Rc::clone(&catalog));
    let mut editor = Editor::new()?;
    editor.set_helper(Some(completer));

    // Determine history file path (default: ~/.nexumdb_history)
    let history_path = dirs::home_dir().map(|h| h.join(".nexumdb_history"));
    if let Some(ref path) = history_path {
        let _ = editor.load_history(path); // Ignore error on first run
    }

    println!();
    print_help_summary();
    println!();

    loop {
        let prompt = format!("{} ", "nexumdb>".bold().cyan());
        match editor.readline(&prompt) {
            Ok(line) => {
                let input = line.trim();

                if input.is_empty() {
                    continue;
                }

                let _ = editor.add_history_entry(input);

                if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
                    println!("{}", "Goodbye! 👋".green().bold());
                    break;
                }

                if input.eq_ignore_ascii_case("help") {
                    print_full_help();
                    continue;
                }

                if input.len() >= 4 && input[..4].eq_ignore_ascii_case("ASK ") {
                    let natural_query = input[4..].trim();

                    if let Some(ref translator) = nl_translator {
                        let schema = get_schema_context(&catalog);

                        let spinner = create_spinner(&format!("Translating: '{}'", natural_query));
                        match translator.translate(natural_query, &schema) {
                            Ok(sql) => {
                                spinner.finish_and_clear();
                                println!("{} {}", "Generated SQL:".cyan().bold(), sql.white());
                                println!();

                                match SqlParser::parse(&sql) {
                                    Ok(statement) => match executor.execute(statement) {
                                        Ok(result) => {
                                            print_result(&result, args.json);
                                        }
                                        Err(e) => {
                                            print_error("Execution error", &e.to_string());
                                        }
                                    },
                                    Err(e) => {
                                        print_error("Parse error", &e.to_string());
                                    }
                                }
                            }
                            Err(e) => {
                                spinner.finish_and_clear();
                                print_error("Translation error", &e.to_string());
                            }
                        }
                    } else {
                        print_error("Error", "Natural language translator not available");
                    }
                    continue;
                }

                // Handle EXPLAIN command
                if input.len() >= 8 && input[..8].eq_ignore_ascii_case("EXPLAIN ") {
                    let query_to_explain = input[8..].trim();

                    if let Some(ref explainer) = query_explainer {
                        println!();
                        match explainer.explain(query_to_explain) {
                            Ok(plan) => {
                                println!("{}", "Query Execution Plan:".cyan().bold());
                                println!("{}", plan);
                            }
                            Err(e) => {
                                print_error("Explain error", &e.to_string());
                            }
                        }
                    } else {
                        print_error("Error", "Query explainer not available");
                    }
                    continue;
                }

                match SqlParser::parse(input) {
                    Ok(statement) => match executor.execute(statement) {
                        Ok(result) => {
                            print_result(&result, args.json);
                        }
                        Err(e) => {
                            print_error("Execution error", &e.to_string());
                        }
                    },
                    Err(e) => {
                        print_error("Parse error", &e.to_string());
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                println!("{}", "Goodbye! 👋".green().bold());
                break;
            }
            Err(err) => {
                print_error("Input error", &err.to_string());
                break;
            }
        }
    }

    // Save history on exit
    if let Some(ref path) = history_path {
        let _ = editor.save_history(path);
    }

    Ok(())
}

fn print_banner() {
    println!();
    println!(
        "{}",
        r#"
  _   _                      ____  ____  
 | \ | | _____  ___   _ _ __ |  _ \| __ ) 
 |  \| |/ _ \ \/ / | | | '_ \| | | |  _ \ 
 | |\  |  __/>  <| |_| | | | | |_| | |_) |
 |_| \_|\___/_/\_\__,_|_| |_|____/|____/ 
"#
        .cyan()
        .bold()
    );
    println!(
        "  {} {}",
        format!("v{}", env!("CARGO_PKG_VERSION")).yellow(),
        "- AI-Native Database with Natural Language Support".white()
    );
    println!();
}

fn print_help_summary() {
    println!("{}", "Quick Commands:".green().bold());
    println!(
        "  {} SQL queries (CREATE, INSERT, SELECT, UPDATE, DELETE, DROP)",
        "•".cyan()
    );
    println!(
        "  {} {} - Natural language queries",
        "•".cyan(),
        "ASK <question>".yellow()
    );
    println!(
        "  {} {} - Show all tables",
        "•".cyan(),
        "SHOW TABLES".yellow()
    );
    println!(
        "  {} {} - Show table structure",
        "•".cyan(),
        "DESCRIBE <table>".yellow()
    );
    println!(
        "  {} {} - Transaction control",
        "•".cyan(),
        "BEGIN / COMMIT / ROLLBACK".yellow()
    );
    println!(
        "  {} {} - Show query plan",
        "•".cyan(),
        "EXPLAIN <query>".yellow()
    );
    println!("  {} {} - Show full help", "•".cyan(), "help".yellow());
    println!("  {} {} - Exit", "•".cyan(), "exit/quit".yellow());
}

fn print_full_help() {
    println!();
    println!("{}", "═".repeat(60).cyan());
    println!("{}", "  NexumDB Command Reference".green().bold());
    println!("{}", "═".repeat(60).cyan());
    println!();

    println!("{}", "SQL Commands:".yellow().bold());
    println!(
        "  {}",
        "CREATE TABLE name (col1 TYPE, col2 TYPE, ...)".white()
    );
    println!(
        "  {}",
        "INSERT INTO name (cols) VALUES (val1, val2, ...)".white()
    );
    println!("  {}", "SELECT * FROM name WHERE condition".white());
    println!(
        "  {}",
        "UPDATE name SET col = value WHERE condition".white()
    );
    println!("  {}", "DELETE FROM name WHERE condition".white());
    println!("  {}", "DROP TABLE [IF EXISTS] name".white());
    println!();

    println!("{}", "Advanced SQL Features:".yellow().bold());
    println!("  {} Pattern matching (%, _)", "LIKE".cyan());
    println!("  {} List membership", "IN".cyan());
    println!("  {} Range queries", "BETWEEN".cyan());
    println!("  {} Multi-column sorting", "ORDER BY".cyan());
    println!("  {} Result truncation", "LIMIT".cyan());
    println!();

    println!("{}", "Special Commands:".yellow().bold());
    println!("  {} - List all tables", "SHOW TABLES".cyan());
    println!("  {} - Show table schema", "DESCRIBE <table>".cyan());
    println!("  {} - Start transaction", "BEGIN".cyan());
    println!("  {} - Commit transaction", "COMMIT".cyan());
    println!("  {} - Roll back transaction", "ROLLBACK".cyan());
    println!("  {} - Natural language query", "ASK <question>".cyan());
    println!("  {} - Query execution plan", "EXPLAIN <query>".cyan());
    println!();

    println!("{}", "Data Types:".yellow().bold());
    println!(
        "  {}, {}, {}, {}",
        "INTEGER".cyan(),
        "FLOAT".cyan(),
        "TEXT".cyan(),
        "BOOLEAN".cyan()
    );
    println!();

    println!("{}", "Examples:".yellow().bold());
    println!(
        "  {}",
        "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)".dimmed()
    );
    println!(
        "  {}",
        "INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Bob', 25)".dimmed()
    );
    println!(
        "  {}",
        "SELECT name, age FROM users WHERE age > 20 ORDER BY name".dimmed()
    );
    println!("  {}", "ASK show me all users older than 25".dimmed());
    println!();
    println!("{}", "═".repeat(60).cyan());
}

fn print_error(prefix: &str, message: &str) {
    eprintln!(
        "{} {}",
        format!("✗ {}:", prefix).red().bold(),
        message.red()
    );
}

fn print_success(message: &str) {
    println!("{} {}", "✓".green().bold(), message.green());
}

fn create_spinner(message: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    spinner.set_message(message.to_string());
    spinner.enable_steady_tick(Duration::from_millis(80));
    spinner
}

fn print_result(result: &ExecutionResult, json_output: bool) {
    if json_output {
        print_result_json(result);
    } else {
        print_result_formatted(result);
    }
}

fn print_result_json(result: &ExecutionResult) {
    let json = match result {
        ExecutionResult::TransactionBegan { tx_id } => {
            serde_json::json!({
                "type": "transaction_began",
                "transaction_id": tx_id,
                "message": format!("Transaction {} started", tx_id)
            })
        }
        ExecutionResult::TransactionCommitted { tx_id, writes } => {
            serde_json::json!({
                "type": "transaction_committed",
                "transaction_id": tx_id,
                "writes": writes,
                "message": format!("Transaction {} committed ({} row mutation(s))", tx_id, writes)
            })
        }
        ExecutionResult::TransactionRolledBack { tx_id } => {
            serde_json::json!({
                "type": "transaction_rolled_back",
                "transaction_id": tx_id,
                "message": format!("Transaction {} rolled back", tx_id)
            })
        }
        ExecutionResult::Created { table } => {
            serde_json::json!({
                "type": "created",
                "table": table,
                "message": format!("Table '{}' created successfully", table)
            })
        }
        ExecutionResult::Inserted { table, rows } => {
            serde_json::json!({
                "type": "inserted",
                "table": table,
                "rows_affected": rows,
                "message": format!("{} row(s) inserted into '{}'", rows, table)
            })
        }
        ExecutionResult::Selected { columns, rows } => {
            let data: Vec<serde_json::Value> = rows
                .iter()
                .map(|row| {
                    let mut obj = serde_json::Map::new();
                    for (i, col) in columns.iter().enumerate() {
                        let value = if i < row.values.len() {
                            value_to_json(&row.values[i])
                        } else {
                            serde_json::Value::Null
                        };
                        obj.insert(col.clone(), value);
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();
            serde_json::json!({
                "type": "selected",
                "columns": columns,
                "row_count": rows.len(),
                "data": data
            })
        }
        ExecutionResult::Updated { table, rows } => {
            serde_json::json!({
                "type": "updated",
                "table": table,
                "rows_affected": rows,
                "message": format!("{} row(s) updated in '{}'", rows, table)
            })
        }
        ExecutionResult::Deleted { table, rows } => {
            serde_json::json!({
                "type": "deleted",
                "table": table,
                "rows_affected": rows,
                "message": format!("{} row(s) deleted from '{}'", rows, table)
            })
        }
        ExecutionResult::TableList { tables } => {
            serde_json::json!({
                "type": "table_list",
                "tables": tables,
                "count": tables.len()
            })
        }
        ExecutionResult::TableDescription { table, columns } => {
            let cols: Vec<serde_json::Value> = columns
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "name": c.name,
                        "type": format!("{:?}", c.data_type)
                    })
                })
                .collect();
            serde_json::json!({
                "type": "table_description",
                "table": table,
                "columns": cols
            })
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_else(|e| {
            serde_json::to_string(&serde_json::json!({ "error": e.to_string() }))
                .unwrap_or_default()
        })
    );
}

fn print_result_formatted(result: &ExecutionResult) {
    match result {
        ExecutionResult::TransactionBegan { tx_id } => {
            print_success(&format!("Transaction {} started", tx_id));
        }
        ExecutionResult::TransactionCommitted { tx_id, writes } => {
            print_success(&format!(
                "Transaction {} committed ({} row mutation(s))",
                tx_id, writes
            ));
        }
        ExecutionResult::TransactionRolledBack { tx_id } => {
            print_success(&format!("Transaction {} rolled back", tx_id));
        }
        ExecutionResult::Created { table } => {
            print_success(&format!("Table '{}' created successfully", table));
        }
        ExecutionResult::Inserted { table, rows } => {
            print_success(&format!("{} row(s) inserted into '{}'", rows, table));
        }
        ExecutionResult::Selected { columns, rows } => {
            if rows.is_empty() {
                println!("{}", "(No rows returned)".dimmed());
                return;
            }

            let mut table = Table::new();
            table.set_content_arrangement(ContentArrangement::Dynamic);

            // Add header row with cyan color
            let header_cells: Vec<Cell> = columns
                .iter()
                .map(|c| Cell::new(c).fg(Color::Cyan))
                .collect();
            table.set_header(header_cells);

            // Add data rows
            for row in rows {
                let row_cells: Vec<Cell> = row
                    .values
                    .iter()
                    .map(|v| Cell::new(format_value(v)))
                    .collect();
                table.add_row(row_cells);
            }

            println!("{table}");
            println!(
                "{}",
                format!(
                    "({} row{})",
                    rows.len(),
                    if rows.len() == 1 { "" } else { "s" }
                )
                .dimmed()
            );
        }
        ExecutionResult::Updated { table, rows } => {
            print_success(&format!("{} row(s) updated in '{}'", rows, table));
        }
        ExecutionResult::Deleted { table, rows } => {
            print_success(&format!("{} row(s) deleted from '{}'", rows, table));
        }
        ExecutionResult::TableList { tables } => {
            if tables.is_empty() {
                println!("{}", "(No tables found)".dimmed());
                return;
            }
            println!("{}", "Tables:".cyan().bold());
            for t in tables {
                println!("  {} {}", "•".green(), t);
            }
        }
        ExecutionResult::TableDescription { table, columns } => {
            println!("{} {}", "Table:".cyan().bold(), table.yellow());

            let mut desc_table = Table::new();
            desc_table.set_content_arrangement(ContentArrangement::Dynamic);
            desc_table.set_header(vec![
                Cell::new("Column").fg(Color::Cyan),
                Cell::new("Type").fg(Color::Cyan),
            ]);

            for col in columns {
                desc_table.add_row(vec![
                    Cell::new(&col.name),
                    Cell::new(format!("{:?}", col.data_type)).fg(Color::Yellow),
                ]);
            }

            println!("{desc_table}");
        }
    }
}

fn format_value(value: &nexum_core::sql::types::Value) -> String {
    value.to_string()
}

fn value_to_json(value: &nexum_core::sql::types::Value) -> serde_json::Value {
    match value {
        nexum_core::sql::types::Value::Integer(i) => serde_json::json!(i),
        nexum_core::sql::types::Value::Float(f) => {
            // Handle NaN and Infinity which are not valid JSON
            if f.is_nan() {
                serde_json::json!("NaN")
            } else if f.is_infinite() {
                if *f > 0.0 {
                    serde_json::json!("Infinity")
                } else {
                    serde_json::json!("-Infinity")
                }
            } else {
                serde_json::json!(f)
            }
        }
        nexum_core::sql::types::Value::Text(t) => serde_json::json!(t),
        nexum_core::sql::types::Value::Boolean(b) => serde_json::json!(b),
        nexum_core::sql::types::Value::Null => serde_json::Value::Null,
    }
}

fn get_schema_context(catalog: &Rc<Catalog>) -> String {
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
