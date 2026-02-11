use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rusqlite::Connection;
use tempfile::{NamedTempFile, tempdir};

use nexum_core::storage::StorageEngine; 
use nexum_core::executor::Executor;
use nexum_core::sql::parser::Parser; 

fn setup_sqlite() -> Connection {
    let db_file = NamedTempFile::new().unwrap();
    let conn = Connection::open(db_file.path()).unwrap();
    conn.execute("CREATE TABLE bench (id INTEGER PRIMARY KEY, val TEXT)", []).unwrap();
    conn
}

fn setup_nexum() -> Executor {
    let db_path = tempdir().unwrap();
    let storage = StorageEngine::new(db_path.path()).unwrap();
    let executor = Executor::new(storage);
    
    let sql = "CREATE TABLE bench (id INTEGER, val TEXT)";
    // Removed .remove(0) because parse returns a Statement directly
    let statement = Parser::parse(sql).unwrap(); 
    executor.execute(statement).unwrap();
    
    executor
}

fn bench_inserts(c: &mut Criterion) {
    let mut group = c.benchmark_group("Insert_Performance");
    
    group.bench_function("SQLite_Single_Insert", |b| {
        let conn = setup_sqlite();
        b.iter(|| {
            conn.execute("INSERT INTO bench (val) VALUES ('test_data')", []).unwrap();
        });
    });

    group.bench_function("NexumDB_Single_Insert", |b| {
        let executor = setup_nexum();
        let sql = "INSERT INTO bench (id, val) VALUES (1, 'test_data')";
        let statement = Parser::parse(sql).unwrap();
        b.iter(|| {
            executor.execute(statement.clone()).unwrap();
        });
    });
    
    group.finish();
}

fn bench_selects(c: &mut Criterion) {
    let mut group = c.benchmark_group("Select_Performance");
    let row_count = 1000;

    let sqlite_conn = setup_sqlite();
    for i in 0..row_count {
        sqlite_conn.execute("INSERT INTO bench (id, val) VALUES (?1, 'data')", [i]).unwrap();
    }

    let nexum_executor = setup_nexum();
    let insert_sql = "INSERT INTO bench (id, val) VALUES (1, 'data')";
    let insert_stmt = Parser::parse(insert_sql).unwrap();
    for _ in 0..row_count {
        nexum_executor.execute(insert_stmt.clone()).unwrap();
    }

    let select_sql = "SELECT val FROM bench WHERE id = 500";
    let select_stmt = Parser::parse(select_sql).unwrap();

    group.bench_function("SQLite_Point_Lookup", |b| {
        b.iter(|| {
            let mut stmt = sqlite_conn.prepare("SELECT val FROM bench WHERE id = 500").unwrap();
            let _ = stmt.query_row([], |r| r.get::<_, String>(0)).unwrap();
        });
    });

    group.bench_function("NexumDB_Point_Lookup_Cold", |b| {
        b.iter(|| {
            black_box(nexum_executor.execute(select_stmt.clone()).unwrap());
        });
    });

    group.bench_function("NexumDB_Point_Lookup_Cached", |b| {
        nexum_executor.execute(select_stmt.clone()).unwrap();
        b.iter(|| {
            black_box(nexum_executor.execute(select_stmt.clone()).unwrap());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_inserts, bench_selects);
criterion_main!(benches);