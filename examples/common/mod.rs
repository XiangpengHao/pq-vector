use arrow::array::{Array, Float32Array, ListArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pq_vector::{IndexBuilder, has_pq_vector_index};
use std::path::Path;

#[allow(unused)]
pub fn read_embedding_at_row(
    path: &Path,
    column_name: &str,
    row: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut current_row = 0;
    for batch in reader {
        let batch = batch?;
        let embedding_col = batch
            .column_by_name(column_name)
            .ok_or("Embedding column not found")?;
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

    Err(format!("Row {row} not found").into())
}

#[allow(unused)]
pub fn ensure_indexed(source: &str, indexed: &str) -> Result<(), Box<dyn std::error::Error>> {
    let indexed_path = Path::new(indexed);
    let needs_build = if indexed_path.exists() {
        !has_pq_vector_index(indexed_path)?
    } else {
        true
    };

    if needs_build {
        println!(
            "Indexed parquet not found or missing pq-vector metadata; building at {indexed}..."
        );
        IndexBuilder::new(source, "embedding").build_new(indexed)?;
    }

    Ok(())
}
