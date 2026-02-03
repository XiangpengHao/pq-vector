use std::fs;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float32Builder, Int32Builder, ListBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn generate_parquet(
    path: &Path,
    rows: usize,
    dim: usize,
    batch_rows: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        ),
    ]));

    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    let mut rng = StdRng::seed_from_u64(1234);

    let mut row = 0usize;
    while row < rows {
        let count = (rows - row).min(batch_rows);
        let mut id_builder = Int32Builder::with_capacity(count);
        let mut list_builder = ListBuilder::new(Float32Builder::with_capacity(count * dim));

        for i in 0..count {
            id_builder.append_value((row + i) as i32);
            for _ in 0..dim {
                list_builder.values().append_value(rng.r#gen::<f32>());
            }
            list_builder.append(true);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(list_builder.finish()),
            ],
        )?;
        writer.write(&batch)?;
        row += count;
    }

    writer.close()?;
    Ok(())
}

#[allow(unused)]
pub fn random_query(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim).map(|_| rng.r#gen::<f32>()).collect()
}

#[allow(unused)]
pub fn array_literal(values: &[f32]) -> String {
    let mut out = String::with_capacity(values.len() * 10);
    out.push('[');
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

pub fn to_mb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}
