use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nexum_core::{Executor, StorageEngine, Parser};
use std::time::Duration;

fn setup_test_data(engine: &StorageEngine, num_records: usize) -> anyhow::Result<()> {
    // Create users table data
    for i in 0..num_records {
        let key = format!("users:{}:id", i);
        engine.set(key.as_bytes(), &i.to_le_bytes())?;
        
        let name_key = format!("users:{}:name", i);
        let name = format!("User{}", i);
        engine.set(name_key.as_bytes(), name.as_bytes())?;
        
        let age_key = format!("users:{}:age", i);
        let age = 20 + (i % 50);
        engine.set(age_key.as_bytes(), &age.to_le_bytes())?;
        
        let email_key = format!("users:{}:email", i);
        let email = format!("user{}@example.com", i);
        engine.set(email_key.as_bytes(), email.as_bytes())?;
    }
    
    // Create products table data
    for i in 0..num_records / 10 {
        let key = format!("products:{}:id", i);
        engine.set(key.as_bytes(), &i.to_le_bytes())?;
        
        let name_key = format!("products:{}:name", i);
        let name = format!("Product{}", i);
        engine.set(name_key.as_bytes(), name.as_bytes())?;
        
        let price_key = format!("products:{}:price", i);
        let price = (10.0 + (i as f64 * 5.5)) as u64;
        engine.set(price_key.as_bytes(), &price.to_le_bytes())?;
    }
    
    Ok(())
}

fn executor_simple_select_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor_simple_select");
    group.measurement_time(Duration::from_secs(10));
    
    for num_records in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*num_records as u64));
        
        group.bench_with_input(
            BenchmarkId::new("select_all", num_records),
            num_records,
            |b, &num_records| {
                b.iter_batched(
                    || {
                        let engine = StorageEngine::memory().unwrap();
                        setup_test_data(&engine, num_records).unwrap();
                        let executor = Executor::new(engine);
                        (executor, "SELECT * FROM users")
                    },
                    |(executor, sql)| {
                        let stmt = Parser::parse(sql).unwrap();
                        black_box(executor.execute(&stmt).unwrap());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

fn executor_filtered_select_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor_filtered_select");
    group.measurement_time(Duration::from_secs(10));
    
    let queries = vec![
        ("age_filter", "SELECT * FROM users WHERE age > 30"),
        ("name_filter", "SELECT * FROM users WHERE name LIKE 'User1%'"),
        ("complex_filter", "SELECT * FROM users WHERE age > 25 AND age < 45"),
        ("multiple_conditions", "SELECT * FROM users WHERE age > 20 AND name LIKE 'User%' AND id < 5000"),
    ];
    
    for (query_name, sql) in queries {
        group.bench_with_input(
            BenchmarkId::new("10k_records", query_name),
            sql,
            |b, sql| {
                b.iter_batched(
                    || {
                        let engine = StorageEngine::memory().unwrap();
                        setup_test_data(&engine, 10000).unwrap();
                        let executor = Executor::new(engine);
                        (executor, sql)
                    },
                    |(executor, sql)| {
                        let stmt = Parser::parse(sql).unwrap();
                        black_box(executor.execute(&stmt).unwrap());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

fn executor_insert_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor_insert");
    
    for batch_size in [1, 10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        // Generate INSERT statement with multiple rows
        let mut values = Vec::new();
        for i in 0..*batch_size {
            values.push(format!("({}, 'NewUser{}', {})", i + 10000, i, 25 + (i % 30)));
        }
        let insert_sql = format!(
            "INSERT INTO users (id, name, age) VALUES {}",
            values.join(", ")
        );
        
        group.bench_with_input(
            BenchmarkId::new("batch_insert", batch_size),
            &insert_sql,
            |b, sql| {
                b.iter_batched(
                    || {
                        let engine = StorageEngine::memory().unwrap();
                        setup_test_data(&engine, 1000).unwrap(); // Pre-populate with some data
                        let executor = Executor::new(engine);
                        (executor, sql)
                    },
                    |(executor, sql)| {
                        let stmt = Parser::parse(sql).unwrap();
                        black_box(executor.execute(&stmt).unwrap());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

fn executor_create_table_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor_create_table");
    
    let table_definitions = vec![
        ("simple_table", "CREATE TABLE simple (id INTEGER, name TEXT)"),
        ("medium_table", "CREATE TABLE medium (id INTEGER, name TEXT, age INTEGER, email TEXT, status TEXT)"),
        ("complex_table", "CREATE TABLE complex (
            id INTEGER,
            first_name TEXT,
            last_name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            zip_code TEXT,
            country TEXT,
            birth_date TEXT,
            registration_date TEXT,
            last_login TEXT,
            status TEXT,
            preferences TEXT
        )"),
    ];
    
    for (table_name, sql) in table_definitions {
        group.bench_function(table_name, |b| {
            b.iter_batched(
                || {
                    let engine = StorageEngine::memory().unwrap();
                    let executor = Executor::new(engine);
                    (executor, sql)
                },
                |(executor, sql)| {
                    let stmt = Parser::parse(sql).unwrap();
                    black_box(executor.execute(&stmt).unwrap());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    
    group.finish();
}

fn executor_mixed_workload_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor_mixed_workload");
    group.measurement_time(Duration::from_secs(15));
    
    let workload = vec![
        "CREATE TABLE test_table (id INTEGER, data TEXT, value INTEGER)",
        "INSERT INTO test_table (id, data, value) VALUES (1, 'test1', 100)",
        "INSERT INTO test_table (id, data, value) VALUES (2, 'test2', 200), (3, 'test3', 300)",
        "SELECT * FROM test_table",
        "SELECT * FROM test_table WHERE value > 150",
        "INSERT INTO test_table (id, data, value) VALUES (4, 'test4', 400)",
        "SELECT id, data FROM test_table WHERE data LIKE 'test%'",
        "SELECT * FROM test_table ORDER BY value DESC",
    ];
    
    group.throughput(Throughput::Elements(workload.len() as u64));
    
    group.bench_function("typical_workload", |b| {
        b.iter_batched(
            || {
                let engine = StorageEngine::memory().unwrap();
                let executor = Executor::new(engine);
                (executor, &workload)
            },
            |(executor, workload)| {
                for sql in workload {
                    let stmt = Parser::parse(sql).unwrap();
                    black_box(executor.execute(&stmt).unwrap());
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

fn executor_large_dataset_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor_large_dataset");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);
    
    for dataset_size in [50000, 100000].iter() {
        group.throughput(Throughput::Elements(*dataset_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("full_table_scan", dataset_size),
            dataset_size,
            |b, &dataset_size| {
                b.iter_batched(
                    || {
                        let engine = StorageEngine::memory().unwrap();
                        setup_test_data(&engine, dataset_size).unwrap();
                        let executor = Executor::new(engine);
                        (executor, "SELECT * FROM users WHERE age > 30")
                    },
                    |(executor, sql)| {
                        let stmt = Parser::parse(sql).unwrap();
                        black_box(executor.execute(&stmt).unwrap());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    executor_simple_select_benchmark,
    executor_filtered_select_benchmark,
    executor_insert_benchmark,
    executor_create_table_benchmark,
    executor_mixed_workload_benchmark,
    executor_large_dataset_benchmark
);
criterion_main!(benches);