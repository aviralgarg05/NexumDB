use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use nexum_core::Parser;

fn parse_create_table_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_parse_create_table");

    let simple_create = "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)";
    let complex_create = "CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        product_name TEXT,
        quantity INTEGER,
        price FLOAT,
        order_date TEXT,
        status TEXT,
        shipping_address TEXT,
        billing_address TEXT,
        notes TEXT
    )";

    group.bench_function("simple_table", |b| {
        b.iter(|| {
            black_box(Parser::parse(simple_create).unwrap());
        });
    });

    group.bench_function("complex_table", |b| {
        b.iter(|| {
            black_box(Parser::parse(complex_create).unwrap());
        });
    });

    group.finish();
}

fn parse_insert_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_parse_insert");

    // Generate INSERT statements with different numbers of rows
    for row_count in [1, 10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*row_count as u64));

        let mut values = Vec::new();
        for i in 0..*row_count {
            values.push(format!("({}, 'User{}', {})", i, i, 20 + (i % 50)));
        }
        let insert_sql = format!(
            "INSERT INTO users (id, name, age) VALUES {}",
            values.join(", ")
        );

        group.bench_with_input(
            BenchmarkId::new("multi_row_insert", row_count),
            &insert_sql,
            |b, sql| {
                b.iter(|| {
                    black_box(Parser::parse(sql).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn parse_select_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_parse_select");

    let queries = vec![
        ("simple_select", "SELECT * FROM users"),
        ("select_with_columns", "SELECT id, name, age FROM users"),
        ("select_with_where", "SELECT * FROM users WHERE age > 25"),
        ("select_with_complex_where", "SELECT id, name FROM users WHERE age > 25 AND name LIKE 'A%' AND id IN (1, 2, 3, 4, 5)"),
        ("select_with_order", "SELECT * FROM users ORDER BY age DESC, name ASC"),
        ("select_with_limit", "SELECT * FROM users WHERE age > 18 ORDER BY name LIMIT 100"),
        ("select_complex", "SELECT id, name, age FROM users WHERE (age BETWEEN 18 AND 65) AND (name LIKE 'John%' OR name LIKE 'Jane%') ORDER BY age DESC, name ASC LIMIT 50"),
    ];

    for (name, sql) in queries {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(Parser::parse(sql).unwrap());
            });
        });
    }

    group.finish();
}

fn parse_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_parse_mixed");

    let statements = vec![
        "CREATE TABLE products (id INTEGER, name TEXT, price FLOAT)",
        "INSERT INTO products (id, name, price) VALUES (1, 'Laptop', 999.99)",
        "SELECT * FROM products WHERE price > 500",
        "INSERT INTO products (id, name, price) VALUES (2, 'Mouse', 29.99), (3, 'Keyboard', 79.99)",
        "SELECT id, name FROM products ORDER BY price DESC",
        "CREATE TABLE categories (id INTEGER, name TEXT, description TEXT)",
        "SELECT * FROM products WHERE name LIKE 'L%'",
        "INSERT INTO categories (id, name, description) VALUES (1, 'Electronics', 'Electronic devices')",
    ];

    group.throughput(Throughput::Elements(statements.len() as u64));

    group.bench_function("mixed_statements", |b| {
        b.iter(|| {
            for stmt in &statements {
                black_box(Parser::parse(stmt).unwrap());
            }
        });
    });

    group.finish();
}

fn parse_error_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_parse_errors");

    let invalid_statements = vec![
        "INVALID SQL STATEMENT",
        "SELECT * FROM",
        "INSERT INTO users VALUES",
        "CREATE TABLE ()",
        "SELECT * FROM users WHERE",
        "",
    ];

    group.bench_function("invalid_sql_handling", |b| {
        b.iter(|| {
            for stmt in &invalid_statements {
                black_box(Parser::parse(stmt).is_err());
            }
        });
    });

    group.finish();
}

fn parse_large_query_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_parse_large");
    group.sample_size(10); // Reduce sample size for large benchmarks

    // Generate a moderately large INSERT statement (reduced from 10k to 1k)
    let mut large_values = Vec::new();
    for i in 0..1000 {
        large_values.push(format!(
            "({}, 'User{}', {}, 'email{}@example.com')",
            i,
            i,
            20 + (i % 50),
            i
        ));
    }
    let large_insert = format!(
        "INSERT INTO users (id, name, age, email) VALUES {}",
        large_values.join(", ")
    );

    group.throughput(Throughput::Bytes(large_insert.len() as u64));

    group.bench_function("large_insert_1k_rows", |b| {
        b.iter(|| {
            black_box(Parser::parse(&large_insert).unwrap());
        });
    });

    // Generate a complex SELECT with many conditions (reduced from 100 to 50)
    let mut conditions = Vec::new();
    for i in 0..50 {
        conditions.push(format!("id = {}", i));
    }
    let large_select = format!("SELECT * FROM users WHERE {}", conditions.join(" OR "));

    group.bench_function("large_select_many_conditions", |b| {
        b.iter(|| {
            black_box(Parser::parse(&large_select).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    parse_create_table_benchmark,
    parse_insert_benchmark,
    parse_select_benchmark,
    parse_mixed_workload,
    parse_error_handling,
    parse_large_query_benchmark
);
criterion_main!(benches);
