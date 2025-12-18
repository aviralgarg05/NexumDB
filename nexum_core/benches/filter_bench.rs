use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nexum_core::executor::filter::ExpressionEvaluator;
use nexum_core::sql::types::Value;
use sqlparser::ast::{Expr, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

fn create_test_row(id: i64, name: &str, age: i64, salary: f64, active: bool) -> Vec<Value> {
    vec![
        Value::Integer(id),
        Value::Text(name.to_string()),
        Value::Integer(age),
        Value::Float(salary),
        Value::Boolean(active),
    ]
}

fn parse_where_clause(sql: &str) -> Expr {
    let dialect = GenericDialect {};
    let ast = Parser::parse_sql(&dialect, &format!("SELECT * FROM t WHERE {}", sql)).unwrap();
    
    if let Statement::Query(query) = &ast[0] {
        if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
            if let Some(where_expr) = &select.selection {
                return where_expr.clone();
            }
        }
    }
    panic!("Failed to parse WHERE clause");
}

fn filter_simple_comparisons_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_simple_comparisons");
    
    let column_names = vec![
        "id".to_string(),
        "name".to_string(),
        "age".to_string(),
        "salary".to_string(),
        "active".to_string(),
    ];
    let evaluator = ExpressionEvaluator::new(column_names);
    
    let test_cases = vec![
        ("integer_eq", "id = 1000", create_test_row(1000, "John", 30, 50000.0, true)),
        ("integer_gt", "age > 25", create_test_row(1, "Jane", 30, 60000.0, true)),
        ("integer_lt", "age < 50", create_test_row(2, "Bob", 25, 45000.0, false)),
        ("float_gt", "salary > 55000.0", create_test_row(3, "Alice", 35, 65000.0, true)),
        ("text_eq", "name = 'John'", create_test_row(4, "John", 28, 52000.0, true)),
        ("boolean_eq", "active = true", create_test_row(5, "Mary", 32, 58000.0, true)),
    ];
    
    for (test_name, sql, row_data) in test_cases {
        let where_expr = parse_where_clause(sql);
        
        group.bench_function(test_name, |b| {
            b.iter(|| {
                black_box(evaluator.evaluate(&where_expr, &row_data).unwrap());
            });
        });
    }
    
    group.finish();
}

fn filter_complex_expressions_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_complex_expressions");
    
    let column_names = vec![
        "id".to_string(),
        "name".to_string(),
        "age".to_string(),
        "salary".to_string(),
        "active".to_string(),
    ];
    let evaluator = ExpressionEvaluator::new(column_names);
    
    let test_cases = vec![
        (
            "and_condition",
            "age > 25 AND salary > 50000.0",
            create_test_row(1, "John", 30, 55000.0, true),
        ),
        (
            "or_condition",
            "age < 25 OR salary > 70000.0",
            create_test_row(2, "Jane", 22, 45000.0, true),
        ),
        (
            "nested_and_or",
            "(age > 30 AND salary > 60000.0) OR (age < 25 AND active = true)",
            create_test_row(3, "Bob", 35, 65000.0, true),
        ),
        (
            "multiple_and",
            "age > 20 AND age < 50 AND salary > 40000.0 AND active = true",
            create_test_row(4, "Alice", 28, 52000.0, true),
        ),
        (
            "complex_nested",
            "((age > 25 AND age < 40) OR salary > 80000.0) AND (name = 'John' OR active = true)",
            create_test_row(5, "John", 32, 58000.0, true),
        ),
    ];
    
    for (test_name, sql, row_data) in test_cases {
        let where_expr = parse_where_clause(sql);
        
        group.bench_function(test_name, |b| {
            b.iter(|| {
                black_box(evaluator.evaluate(&where_expr, &row_data).unwrap());
            });
        });
    }
    
    group.finish();
}

fn filter_like_patterns_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_like_patterns");
    
    let column_names = vec!["name".to_string(), "email".to_string()];
    let evaluator = ExpressionEvaluator::new(column_names);
    
    let test_cases = vec![
        (
            "prefix_match",
            "name LIKE 'John%'",
            vec![Value::Text("John Smith".to_string()), Value::Text("john@example.com".to_string())],
        ),
        (
            "suffix_match",
            "email LIKE '%@gmail.com'",
            vec![Value::Text("Alice Johnson".to_string()), Value::Text("alice@gmail.com".to_string())],
        ),
        (
            "contains_match",
            "name LIKE '%son%'",
            vec![Value::Text("Johnson".to_string()), Value::Text("johnson@example.com".to_string())],
        ),
        (
            "single_char_wildcard",
            "name LIKE 'J_hn'",
            vec![Value::Text("John".to_string()), Value::Text("john@example.com".to_string())],
        ),
        (
            "complex_pattern",
            "email LIKE '%@%.com'",
            vec![Value::Text("Bob Wilson".to_string()), Value::Text("bob@company.com".to_string())],
        ),
    ];
    
    for (test_name, sql, row_data) in test_cases {
        let where_expr = parse_where_clause(sql);
        
        group.bench_function(test_name, |b| {
            b.iter(|| {
                black_box(evaluator.evaluate(&where_expr, &row_data).unwrap());
            });
        });
    }
    
    group.finish();
}

fn filter_in_list_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_in_list");
    
    let column_names = vec!["id".to_string(), "status".to_string()];
    let evaluator = ExpressionEvaluator::new(column_names);
    
    for list_size in [5, 10, 50, 100].iter() {
        let mut id_list = Vec::new();
        for i in 0..*list_size {
            id_list.push(i.to_string());
        }
        let sql = format!("id IN ({})", id_list.join(", "));
        let where_expr = parse_where_clause(&sql);
        
        let row_data = vec![
            Value::Integer(*list_size / 2), // Should be found in the middle
            Value::Text("active".to_string()),
        ];
        
        group.bench_with_input(
            BenchmarkId::new("id_in_list", list_size),
            &where_expr,
            |b, expr| {
                b.iter(|| {
                    black_box(evaluator.evaluate(expr, &row_data).unwrap());
                });
            },
        );
    }
    
    // Test string IN lists
    let status_sql = "status IN ('active', 'pending', 'inactive', 'suspended', 'archived')";
    let status_expr = parse_where_clause(status_sql);
    let status_row = vec![
        Value::Integer(1),
        Value::Text("active".to_string()),
    ];
    
    group.bench_function("string_in_list", |b| {
        b.iter(|| {
            black_box(evaluator.evaluate(&status_expr, &status_row).unwrap());
        });
    });
    
    group.finish();
}

fn filter_between_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_between");
    
    let column_names = vec!["age".to_string(), "salary".to_string(), "score".to_string()];
    let evaluator = ExpressionEvaluator::new(column_names);
    
    let test_cases = vec![
        (
            "integer_between",
            "age BETWEEN 25 AND 65",
            vec![Value::Integer(35), Value::Float(55000.0), Value::Float(85.5)],
        ),
        (
            "float_between",
            "salary BETWEEN 40000.0 AND 80000.0",
            vec![Value::Integer(30), Value::Float(65000.0), Value::Float(92.3)],
        ),
        (
            "score_between",
            "score BETWEEN 80.0 AND 100.0",
            vec![Value::Integer(28), Value::Float(52000.0), Value::Float(88.7)],
        ),
    ];
    
    for (test_name, sql, row_data) in test_cases {
        let where_expr = parse_where_clause(sql);
        
        group.bench_function(test_name, |b| {
            b.iter(|| {
                black_box(evaluator.evaluate(&where_expr, &row_data).unwrap());
            });
        });
    }
    
    group.finish();
}

fn filter_batch_evaluation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_batch_evaluation");
    
    let column_names = vec![
        "id".to_string(),
        "name".to_string(),
        "age".to_string(),
        "salary".to_string(),
        "active".to_string(),
    ];
    let evaluator = ExpressionEvaluator::new(column_names);
    
    // Create test data
    let mut test_rows = Vec::new();
    for i in 0..10000 {
        test_rows.push(create_test_row(
            i,
            &format!("User{}", i),
            20 + (i % 50),
            30000.0 + (i as f64 * 5.0),
            i % 2 == 0,
        ));
    }
    
    let filters = vec![
        ("simple_filter", "age > 30"),
        ("complex_filter", "age > 25 AND salary > 50000.0 AND active = true"),
        ("like_filter", "name LIKE 'User1%'"),
        ("range_filter", "age BETWEEN 25 AND 45 AND salary BETWEEN 40000.0 AND 70000.0"),
    ];
    
    for (filter_name, sql) in filters {
        let where_expr = parse_where_clause(sql);
        
        group.throughput(Throughput::Elements(test_rows.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("10k_rows", filter_name),
            &where_expr,
            |b, expr| {
                b.iter(|| {
                    let mut matches = 0;
                    for row in &test_rows {
                        if evaluator.evaluate(expr, row).unwrap_or(false) {
                            matches += 1;
                        }
                    }
                    black_box(matches);
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    filter_simple_comparisons_benchmark,
    filter_complex_expressions_benchmark,
    filter_like_patterns_benchmark,
    filter_in_list_benchmark,
    filter_between_benchmark,
    filter_batch_evaluation_benchmark
);
criterion_main!(benches);