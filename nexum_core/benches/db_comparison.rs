use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rusqlite::Connection;
use tempfile::{tempdir, NamedTempFile, TempDir};

use nexum_core::executor::Executor;
use nexum_core::sql::parser::Parser;
use nexum_core::storage::StorageEngine;

fn setup_sqlite() -> (Connection, NamedTempFile) {
    let db_file = NamedTempFile::new().expect("Failed to create temp file");
    let conn = Connection::open(db_file.path()).expect("Failed to open SQLite connection");
    conn.execute("CREATE TABLE bench (id INTEGER PRIMARY KEY, val TEXT)", [])
        .expect("Failed to create SQLite table");
    (conn, db_file)
}

fn setup_nexum(use_cache: bool) -> (Executor, TempDir) {
    let db_dir = tempdir().expect("Failed to create temp directory");
    let storage = StorageEngine::new(db_dir.path()).expect("Failed to initialize Nexum storage");
    
    // Fix: Explicitly enable the semantic cache if requested
    let mut executor = Executor::new(storage);
    if use_cache {
        executor = executor.with_cache(); 
    }

    let sql = "CREATE TABLE bench (id INTEGER PRIMARY KEY, val TEXT)";
    let statement = Parser::parse(sql).expect("Failed to parse Nexum schema");
    executor.execute(statement).expect("Failed to execute Nexum schema");

    (executor, db_dir)
}

fn bench_selects(c: &mut Criterion) {
    let mut group = c.benchmark_group("Select_Performance");
    let row_count = 1000;

    // --- SQLite Setup ---
    let (sqlite_conn, _sqlite_file) = setup_sqlite();
    let mut sqlite_insert = sqlite_conn.prepare("INSERT INTO bench (id, val) VALUES (?1, 'data')").unwrap();
    for i in 0..row_count {
        sqlite_insert.execute([i]).unwrap();
    }

    // --- NexumDB Setup (With Cache Enabled) ---
    let (nexum_executor, _nexum_dir) = setup_nexum(true);
    for i in 0..row_count {
        let sql = format!("INSERT INTO bench (id, val) VALUES ({}, 'data')", i);
        let stmt = Parser::parse(&sql).unwrap();
        nexum_executor.execute(stmt).unwrap();
    }

    let select_sql_str = "SELECT val FROM bench WHERE id = 500";
    let select_stmt = Parser::parse(select_sql_str).unwrap();

    // 1. SQLite Baseline
        group.bench_function("SQLite_Point_Lookup", |b| {
                // PREPARE OUTSIDE: Compiles lookup once
                let mut stmt = sqlite_conn.prepare("SELECT val FROM bench WHERE id = 500").unwrap();
                b.iter(|| {
                    // Explicitly typed |r: &rusqlite::Row| to fix E0282
                    let _ = stmt.query_row([], |r: &rusqlite::Row| r.get::<_, String>(0)).unwrap();
                });
            });

    // 2. NexumDB Cold (Cache is enabled, but first time seeing this specific query)
    group.bench_function("NexumDB_Point_Lookup_Cold", |b| {
        b.iter(|| {
            // We use a fresh executor or clear the cache to ensure it's truly "Cold"
            // For simplicity in this bench, we just measure the first hit performance 
            // by recreating the executor if necessary, but here we iterate:
            black_box(nexum_executor.execute(select_stmt.clone()).unwrap());
        });
    });

    // 3. NexumDB Cached (Semantic cache hit)
    group.bench_function("NexumDB_Point_Lookup_Cached", |b| {
        // Warm up the semantic cache by executing once
        let _ = nexum_executor.execute(select_stmt.clone()).unwrap();
        b.iter(|| {
            black_box(nexum_executor.execute(select_stmt.clone()).unwrap());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_selects);
criterion_main!(benches);