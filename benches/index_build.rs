mod bench_util;

use mimalloc::MiMalloc;
use std::fs;
use std::path::Path;
use std::time::Instant;

use pq_vector::IndexBuilder;

use bench_util::{generate_parquet, to_mb};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const ROWS: usize = 1_000_000;
const DIM: usize = 1024;
const BATCH_ROWS: usize = 2048;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = Path::new("data");
    fs::create_dir_all(data_dir)?;

    let source_path = data_dir.join("index_build_source.parquet");
    let work_path = data_dir.join("index_build_work.parquet");

    if !source_path.exists() {
        println!("=== Generating synthetic dataset ===");
        println!("rows={ROWS}, dim={DIM}, batch_rows={BATCH_ROWS}");

        let gen_start = Instant::now();
        generate_parquet(&source_path, ROWS, DIM, BATCH_ROWS)?;
        let gen_time = gen_start.elapsed();
        let size = fs::metadata(&source_path)?.len();
        println!("Generated parquet in {:.2?}", gen_time);
        println!("Source parquet size: {:.2} MB", to_mb(size));
    }

    if work_path.exists() {
        fs::remove_file(&work_path)?;
    }
    fs::copy(&source_path, &work_path)?;

    println!("\n=== Building IVF index (in-place, defaults) ===");
    let index_start = Instant::now();
    IndexBuilder::new(&work_path, "embedding").build_inplace()?;
    let index_time = index_start.elapsed();

    let source_size = fs::metadata(&source_path)?.len();
    let indexed_size = fs::metadata(&work_path)?.len();
    println!("Index build time: {:.2?}", index_time);
    println!("Indexed parquet size: {:.2} MB", to_mb(indexed_size));
    println!(
        "Index overhead: {:.2} MB ({:.1}%)",
        to_mb(indexed_size - source_size),
        (indexed_size - source_size) as f64 / source_size as f64 * 100.0
    );

    Ok(())
}
