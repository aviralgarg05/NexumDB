use nexum_core::{Executor, Parser, StorageEngine, sql::types::{Value, ExecutionResult}};

#[test]
fn test_count_aggregate() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE items (id INTEGER, name TEXT)").unwrap();
    executor.execute(create).unwrap();

    let insert = Parser::parse(
        "INSERT INTO items (id, name) VALUES (1, 'A'), (2, 'B'), (3, NULL)",
    ).unwrap();
    executor.execute(insert).unwrap();

    // Test COUNT(*)
    let select = Parser::parse("SELECT COUNT(*) FROM items").unwrap();
    let result = executor.execute(select).unwrap();
    match result {
        ExecutionResult::Selected { rows, .. } => {
             if let Value::Integer(count) = rows[0].values[0] {
                assert_eq!(count, 3);
            } else {
                panic!("Expected integer count");
            }
        }
        _ => panic!("Expected Selected result"),
    }

     // Test COUNT(column)
    let select = Parser::parse("SELECT COUNT(name) FROM items").unwrap();
    let result = executor.execute(select).unwrap();
    match result {
        ExecutionResult::Selected { rows, .. } => {
             if let Value::Integer(count) = rows[0].values[0] {
                assert_eq!(count, 2); // Should ignore NULL
            } else {
                panic!("Expected integer count");
            }
        }
        _ => panic!("Expected Selected result"),
    }
}

#[test]
fn test_sum_avg_aggregate() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE sales (amount INTEGER, value FLOAT)").unwrap();
    executor.execute(create).unwrap();

    let insert = Parser::parse(
        "INSERT INTO sales (amount, value) VALUES (10, 1.5), (20, 2.5), (30, 3.5)",
    ).unwrap();
    executor.execute(insert).unwrap();

    // Test SUM(amount) - Integer
    let select = Parser::parse("SELECT SUM(amount) FROM sales").unwrap();
    let result = executor.execute(select).unwrap();
    match result {
        ExecutionResult::Selected { rows, .. } => {
             if let Value::Integer(sum) = rows[0].values[0] {
                assert_eq!(sum, 60);
            } else {
                panic!("Expected integer sum");
            }
        }
        _ => panic!("Expected Selected result");
    }

    // Test SUM(value) - Float
    let select = Parser::parse("SELECT SUM(value) FROM sales").unwrap();
    let result = executor.execute(select).unwrap();
    match result {
        ExecutionResult::Selected { rows, .. } => {
             if let Value::Float(sum) = rows[0].values[0] {
                assert!((sum - 7.5).abs() < f64::EPSILON);
            } else {
                panic!("Expected float sum");
            }
        }
        _ => panic!("Expected Selected result");
    }

     // Test AVG(amount)
    let select = Parser::parse("SELECT AVG(amount) FROM sales").unwrap();
    let result = executor.execute(select).unwrap();
    match result {
        ExecutionResult::Selected { rows, .. } => {
             if let Value::Float(avg) = rows[0].values[0] {
                assert!((avg - 20.0).abs() < f64::EPSILON);
            } else {
                panic!("Expected float avg, got {:?}", rows[0].values[0]);
            }
        }
        _ => panic!("Expected Selected result");
    }
}

#[test]
fn test_min_max_aggregate() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE scores (score INTEGER)").unwrap();
    executor.execute(create).unwrap();

    let insert = Parser::parse(
        "INSERT INTO scores (score) VALUES (10), (50), (30), (5)",
    ).unwrap();
    executor.execute(insert).unwrap();

    // Test MIN
    let select = Parser::parse("SELECT MIN(score) FROM scores").unwrap();
    let result = executor.execute(select).unwrap();
    match result {
        ExecutionResult::Selected { rows, .. } => {
             if let Value::Integer(min) = rows[0].values[0] {
                assert_eq!(min, 5);
            } else {
                panic!("Expected integer min");
            }
        }
        _ => panic!("Expected Selected result");
    }

    // Test MAX
    let select = Parser::parse("SELECT MAX(score) FROM scores").unwrap();
    let result = executor.execute(select).unwrap();
    match result {
        ExecutionResult::Selected { rows, .. } => {
             if let Value::Integer(max) = rows[0].values[0] {
                assert_eq!(max, 50);
            } else {
                panic!("Expected integer max");
            }
        }
        _ => panic!("Expected Selected result");
    }
}

#[test]
fn test_mixed_aggregates() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE data (val INTEGER)").unwrap();
    executor.execute(create).unwrap();

    let insert = Parser::parse(
        "INSERT INTO data (val) VALUES (1), (2), (3)",
    ).unwrap();
    executor.execute(insert).unwrap();
    
    // SELECT COUNT(*), SUM(val), MAX(val)
    let select = Parser::parse("SELECT COUNT(*), SUM(val), MAX(val) FROM data").unwrap();
    let result = executor.execute(select).unwrap();
     match result {
        ExecutionResult::Selected { rows, columns } => {
            assert_eq!(rows.len(), 1);
            assert_eq!(columns.len(), 3);
            
            // COUNT(*)
            if let Value::Integer(c) = rows[0].values[0] {
                assert_eq!(c, 3);
            } else { panic!("Wrong type"); }

             // SUM(val)
            if let Value::Integer(s) = rows[0].values[1] {
                assert_eq!(s, 6);
            } else { panic!("Wrong type"); }
            
             // MAX(val)
            if let Value::Integer(m) = rows[0].values[2] {
                assert_eq!(m, 3);
            } else { panic!("Wrong type"); }
        }
        _ => panic!("Expected Selected result");
    }
}
