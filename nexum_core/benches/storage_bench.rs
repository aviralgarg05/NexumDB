use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use nexum_core::StorageEngine;

fn storage_write_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_write");

    // Test different data sizes
    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential_writes", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || StorageEngine::memory().unwrap(),
                    |engine| {
                        for i in 0..size {
                            let key = format!("key_{:06}", i);
                            let value = format!("value_{:06}_data_payload", i);
                            engine.set(key.as_bytes(), value.as_bytes()).unwrap();
                        }
                        black_box(())
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn storage_read_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_read");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential_reads", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let engine = StorageEngine::memory().unwrap();
                        // Pre-populate with data
                        for i in 0..size {
                            let key = format!("key_{:06}", i);
                            let value = format!("value_{:06}_data_payload", i);
                            engine.set(key.as_bytes(), value.as_bytes()).unwrap();
                        }
                        engine
                    },
                    |engine| {
                        for i in 0..size {
                            let key = format!("key_{:06}", i);
                            black_box(engine.get(key.as_bytes()).unwrap());
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn storage_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_mixed");
    group.measurement_time(Duration::from_secs(10));

    for ratio in [(70, 30), (50, 50), (30, 70)].iter() {
        let (read_pct, write_pct) = ratio;

        group.bench_with_input(
            BenchmarkId::new("read_write_mix", format!("{}r_{}w", read_pct, write_pct)),
            ratio,
            |b, &(read_pct, _write_pct)| {
                b.iter_batched(
                    || {
                        let engine = StorageEngine::memory().unwrap();
                        // Pre-populate with some data
                        for i in 0..1000 {
                            let key = format!("key_{:06}", i);
                            let value = format!("value_{:06}", i);
                            engine.set(key.as_bytes(), value.as_bytes()).unwrap();
                        }
                        engine
                    },
                    |engine| {
                        for i in 0..1000 {
                            if i % 100 < read_pct {
                                // Read operation
                                let key = format!("key_{:06}", i % 1000);
                                black_box(engine.get(key.as_bytes()).unwrap());
                            } else {
                                // Write operation
                                let key = format!("key_{:06}", i + 1000);
                                let value = format!("new_value_{:06}", i);
                                engine.set(key.as_bytes(), value.as_bytes()).unwrap();
                            }
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn storage_scan_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_scan");

    for prefix_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("prefix_scan", prefix_size),
            prefix_size,
            |b, &prefix_size| {
                b.iter_batched(
                    || {
                        let engine = StorageEngine::memory().unwrap();
                        // Create data with different prefixes
                        for prefix in 0..10 {
                            for i in 0..prefix_size {
                                let key = format!("prefix_{}:item_{:06}", prefix, i);
                                let value = format!("data_{}", i);
                                engine.set(key.as_bytes(), value.as_bytes()).unwrap();
                            }
                        }
                        engine
                    },
                    |engine| {
                        black_box(engine.scan_prefix(b"prefix_5:").unwrap());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn storage_persistence_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_persistence");

    group.bench_function("flush_performance", |b| {
        b.iter_batched(
            || {
                let temp_dir = tempfile::tempdir().unwrap();
                let engine = StorageEngine::new(temp_dir.path().join("bench_db")).unwrap();

                // Write some data
                for i in 0..1000 {
                    let key = format!("key_{:06}", i);
                    let value = format!("value_{:06}", i);
                    engine.set(key.as_bytes(), value.as_bytes()).unwrap();
                }

                (engine, temp_dir)
            },
            |(engine, _temp_dir)| {
                engine.flush().unwrap();
                black_box(())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    storage_write_throughput,
    storage_read_throughput,
    storage_mixed_workload,
    storage_scan_prefix,
    storage_persistence_benchmark
);
criterion_main!(benches);
