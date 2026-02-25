use nexum_core::{Executor, executor::ExecutionResult, Parser, StorageEngine, sql::types::Value};

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
        _ => panic!("Expected Selected result"),
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
        _ => panic!("Expected Selected result"),
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
        _ => panic!("Expected Selected result"),
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
        _ => panic!("Expected Selected result"),
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
        _ => panic!("Expected Selected result"),
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
        _ => panic!("Expected Selected result"),
    }
}

#[test]
fn test_empty_table_aggregates() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE empty (id INTEGER)").unwrap();
    executor.execute(create).unwrap();

    // COUNT(*) should be 0
    let select = Parser::parse("SELECT COUNT(*) FROM empty").unwrap();
    if let ExecutionResult::Selected { rows, .. } = executor.execute(select).unwrap() {
        assert_eq!(rows[0].values[0], Value::Integer(0));
    } else {
        panic!("Expected Selected result");
    }

    // MIN/MAX/SUM/AVG should be NULL
    let select = Parser::parse("SELECT MIN(id), MAX(id), SUM(id), AVG(id) FROM empty").unwrap();
    if let ExecutionResult::Selected { rows, .. } = executor.execute(select).unwrap() {
        let vals = &rows[0].values;
        assert!(matches!(vals[0], Value::Null));
        assert!(matches!(vals[1], Value::Null));
        assert!(matches!(vals[2], Value::Null));
        assert!(matches!(vals[3], Value::Null));
    } else {
        panic!("Expected Selected result");
    }
}

#[test]
fn test_null_only_aggregates() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE nulls (val INTEGER)").unwrap();
    executor.execute(create).unwrap();

    let insert = Parser::parse("INSERT INTO nulls (val) VALUES (NULL), (NULL)").unwrap();
    executor.execute(insert).unwrap();

     // SUM/AVG should be NULL
    let select = Parser::parse("SELECT SUM(val), AVG(val) FROM nulls").unwrap();
    if let ExecutionResult::Selected { rows, .. } = executor.execute(select).unwrap() {
        let vals = &rows[0].values;
         assert!(matches!(vals[0], Value::Null));
         assert!(matches!(vals[1], Value::Null));
    } else {
        panic!("Expected Selected result");
    }
}

#[test]
fn test_min_max_types() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);

    let create = Parser::parse("CREATE TABLE mixed (f FLOAT, t TEXT)").unwrap();
    executor.execute(create).unwrap();

    let insert = Parser::parse(
        "INSERT INTO mixed (f, t) VALUES (1.5, 'a'), (2.5, 'b'), (0.5, 'c')",
    ).unwrap();
    executor.execute(insert).unwrap();

    // MIN(f)
    let select = Parser::parse("SELECT MIN(f) FROM mixed").unwrap();
    if let ExecutionResult::Selected { rows, .. } = executor.execute(select).unwrap() {
         if let Value::Float(f) = rows[0].values[0] {
            assert!((f - 0.5).abs() < f64::EPSILON);
        } else { panic!("Expected Float"); }
    }

    // MAX(t)
    let select = Parser::parse("SELECT MAX(t) FROM mixed").unwrap();
    if let ExecutionResult::Selected { rows, .. } = executor.execute(select).unwrap() {
         if let Value::Text(ref t) = rows[0].values[0] {
            assert_eq!(t, "c");
        } else { panic!("Expected Text"); }
    }
}

