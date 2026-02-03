use arrow::array::{Array, Float32Array, ListArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pq_vector::{IndexBuilder, TopkBuilder};
use std::fs::File;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source_path = Path::new("data/vldb_2025.parquet");
    let output_path = Path::new("data/vldb_2025_indexed.parquet");
    let embedding_column = "embedding";

    // Build the index and embed it in a new parquet file
    println!("=== Building IVF Index ===\n");
    IndexBuilder::new(source_path, embedding_column)
        .max_iters(10)
        .seed(42)
        .build_new(output_path)?;

    // Get a query vector (use the first embedding from the file)
    let query = get_embedding_at_row(output_path, embedding_column, 0)?;
    println!("\n=== Searching for top 10 similar papers ===");
    println!("Query: first paper's embedding (should return itself as top result)\n");

    // Search with different nprobe values
    for nprobe in [1, 3, 5, 10] {
        let results = TopkBuilder::new(output_path, &query)
            .k(10)?
            .nprobe(nprobe)?
            .search()
            .await?;
        println!(
            "nprobe={}: top results: {:?}",
            nprobe,
            results
                .iter()
                .take(3)
                .map(|r| (r.row_idx, format!("{:.4}", r.distance)))
                .collect::<Vec<_>>()
        );
    }

    // Show detailed results
    println!("\n=== Detailed Results (nprobe=5) ===\n");
    let results = TopkBuilder::new(output_path, &query)
        .k(5)?
        .nprobe(5)?
        .search()
        .await?;
    let titles = get_titles(output_path)?;

    for (i, result) in results.iter().enumerate() {
        let title = &titles[result.row_idx as usize];
        println!(
            "{}. [row {}] distance={:.4}",
            i + 1,
            result.row_idx,
            result.distance
        );
        println!("   {}\n", title);
    }

    // Test with a different query
    println!("=== Searching with row 100's embedding ===\n");
    let query2 = get_embedding_at_row(output_path, embedding_column, 100)?;
    let results = TopkBuilder::new(output_path, &query2)
        .k(5)?
        .nprobe(5)?
        .search()
        .await?;

    for (i, result) in results.iter().enumerate() {
        let title = &titles[result.row_idx as usize];
        println!(
            "{}. [row {}] distance={:.4}",
            i + 1,
            result.row_idx,
            result.distance
        );
        println!("   {}\n", title);
    }

    Ok(())
}

fn get_embedding_at_row(
    path: &Path,
    column_name: &str,
    row: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut current_row = 0;
    for batch in reader {
        let batch = batch?;
        let embedding_col = batch.column_by_name(column_name).unwrap();
        let list_array = embedding_col.as_any().downcast_ref::<ListArray>().unwrap();

        if row < current_row + list_array.len() {
            let local_row = row - current_row;
            let start = list_array.value_offsets()[local_row] as usize;
            let end = list_array.value_offsets()[local_row + 1] as usize;
            let values = list_array.values();
            let float_array = values.as_any().downcast_ref::<Float32Array>().unwrap();
            return Ok((start..end).map(|i| float_array.value(i)).collect());
        }
        current_row += list_array.len();
    }
    Err("Row not found".into())
}

fn get_titles(path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut titles = Vec::new();
    for batch in reader {
        let batch = batch?;
        let title_col = batch.column_by_name("title").unwrap();
        let string_array = title_col.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..string_array.len() {
            titles.push(string_array.value(i).to_string());
        }
    }
    Ok(titles)
}
