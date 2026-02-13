use nexum_core::{Executor, Parser, StorageEngine};

#[test]
fn test_advanced_sql_features() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create =
        Parser::parse("CREATE TABLE items (id INTEGER, name TEXT, price INTEGER)").unwrap();
    executor.execute(create).unwrap();

    let insert = Parser::parse(
        "INSERT INTO items (id, name, price) VALUES (1, 'TestA', 100), (2, 'TestB', 200), (3, 'OtherC', 50), (4, 'TestC', 150)"
    ).unwrap();
    executor.execute(insert).unwrap();

    let select =
        Parser::parse("SELECT * FROM items WHERE name LIKE 'Test%' ORDER BY price DESC LIMIT 2")
            .unwrap();
    let result = executor.execute(select).unwrap();

    match result {
        nexum_core::executor::ExecutionResult::Selected { rows, .. } => {
            println!("Result rows: {:?}", rows);
            assert_eq!(rows.len(), 2);
        }
        _ => panic!("Expected Selected result"),
    }
}

#[test]
fn test_in_operator_integration() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE orders (id INTEGER, status TEXT)").unwrap();
    executor.execute(create).unwrap();

    let insert = Parser::parse(
        "INSERT INTO orders VALUES (1, 'active'), (2, 'pending'), (3, 'completed'), (4, 'active')",
    )
    .unwrap();
    executor.execute(insert).unwrap();

    let select =
        Parser::parse("SELECT * FROM orders WHERE status IN ('active', 'pending')").unwrap();
    let result = executor.execute(select).unwrap();

    match result {
        nexum_core::executor::ExecutionResult::Selected { rows, .. } => {
            assert_eq!(rows.len(), 3);
        }
        _ => panic!("Expected Selected result"),
    }
}

#[test]
fn test_between_with_order_limit() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE products (id INTEGER, price INTEGER)").unwrap();
    executor.execute(create).unwrap();

    let insert = Parser::parse(
        "INSERT INTO products VALUES (1, 50), (2, 150), (3, 250), (4, 175), (5, 125)",
    )
    .unwrap();
    executor.execute(insert).unwrap();

    let select = Parser::parse(
        "SELECT * FROM products WHERE price BETWEEN 100 AND 200 ORDER BY price ASC LIMIT 3",
    )
    .unwrap();
    let result = executor.execute(select).unwrap();

    match result {
        nexum_core::executor::ExecutionResult::Selected { rows, .. } => {
            assert_eq!(rows.len(), 3);
        }
        _ => panic!("Expected Selected result"),
    }
}

#[test]
fn test_table_lifecycle_and_projection() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)").unwrap();
    executor.execute(create).unwrap();

    let insert =
        Parser::parse("INSERT INTO users (name, id) VALUES ('Alice', 1), ('Bob', 2)").unwrap();
    executor.execute(insert).unwrap();

    let select = Parser::parse("SELECT name AS display_name FROM users ORDER BY id ASC").unwrap();
    let result = executor.execute(select).unwrap();
    match result {
        nexum_core::executor::ExecutionResult::Selected { rows, columns } => {
            assert_eq!(columns, vec!["display_name".to_string()]);
            assert_eq!(rows.len(), 2);
            assert_eq!(
                rows[0].values[0],
                nexum_core::sql::types::Value::Text("Alice".to_string())
            );
            assert_eq!(
                rows[1].values[0],
                nexum_core::sql::types::Value::Text("Bob".to_string())
            );
        }
        _ => panic!("Expected Selected result"),
    }

    let show = Parser::parse("SHOW TABLES").unwrap();
    let result = executor.execute(show).unwrap();
    match result {
        nexum_core::executor::ExecutionResult::TableList { tables } => {
            assert_eq!(tables, vec!["users".to_string()]);
        }
        _ => panic!("Expected TableList result"),
    }

    let describe = Parser::parse("DESCRIBE users").unwrap();
    let result = executor.execute(describe).unwrap();
    match result {
        nexum_core::executor::ExecutionResult::TableDescription { table, columns } => {
            assert_eq!(table, "users");
            assert_eq!(columns.len(), 3);
        }
        _ => panic!("Expected TableDescription result"),
    }

    let drop = Parser::parse("DROP TABLE users").unwrap();
    executor.execute(drop).unwrap();

    let show = Parser::parse("SHOW TABLES").unwrap();
    let result = executor.execute(show).unwrap();
    match result {
        nexum_core::executor::ExecutionResult::TableList { tables } => {
            assert!(tables.is_empty());
        }
        _ => panic!("Expected TableList result"),
    }
}

#[test]
fn test_transaction_commands_flow() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE ledger (id INTEGER, amount INTEGER)").unwrap();
    executor.execute(create).unwrap();

    let begin = Parser::parse("BEGIN").unwrap();
    executor.execute(begin).unwrap();

    let insert = Parser::parse("INSERT INTO ledger VALUES (1, 100)").unwrap();
    executor.execute(insert).unwrap();

    let rollback = Parser::parse("ROLLBACK").unwrap();
    executor.execute(rollback).unwrap();

    let select = Parser::parse("SELECT * FROM ledger").unwrap();
    let result = executor.execute(select).unwrap();
    match result {
        nexum_core::executor::ExecutionResult::Selected { rows, .. } => {
            assert_eq!(rows.len(), 0);
        }
        _ => panic!("Expected Selected result"),
    }

    let begin = Parser::parse("BEGIN TRANSACTION").unwrap();
    executor.execute(begin).unwrap();
    let insert = Parser::parse("INSERT INTO ledger VALUES (2, 250)").unwrap();
    executor.execute(insert).unwrap();
    let commit = Parser::parse("COMMIT").unwrap();
    executor.execute(commit).unwrap();

    let select = Parser::parse("SELECT * FROM ledger").unwrap();
    let result = executor.execute(select).unwrap();
    match result {
        nexum_core::executor::ExecutionResult::Selected { rows, .. } => {
            assert_eq!(rows.len(), 1);
        }
        _ => panic!("Expected Selected result"),
    }
}
