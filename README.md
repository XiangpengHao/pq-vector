# pq-vector

[![CI](https://github.com/XiangpengHao/pq-vector/actions/workflows/ci.yml/badge.svg)](https://github.com/XiangpengHao/pq-vector/actions/workflows/ci.yml)

Vector Search with only Parquet and DataFusion

## Features

- **Embedded Index**: Index stored within the Parquet file itself - no separate index files
- **Standard Compatible**: Indexed files remain valid Parquet - DuckDB, Pandas, etc. can read them normally
- **DataFusion integration**: Ergonomic vector search with just SQL.
- **Zero-copy**: Zero-copy, in-place Parquet indexing.

## Quick start

### 1) Build an index

```rust
use pq_vector::{EmbeddingColumn, IndexBuilder, IvfBuildParams};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    IndexBuilder::new(
        Path::new("data/embeddings.parquet"),         // Source file
        Path::new("data/embeddings_indexed.parquet"), // Output file
        EmbeddingColumn::try_from("embedding")?,      // Column name containing vectors
    )
    .params(IvfBuildParams::default())
    .build()?;

    Ok(())
}
```

### 2) Search with Rust

```rust
use pq_vector::TopkBuilder;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query_vector: Vec<f32> = vec![/* your query embedding */];

    let results = TopkBuilder::new(
        Path::new("data/embeddings_indexed.parquet"),
        &query_vector,
    )
    .k(10)?
    .nprobe(5)?
    .search()
    .await?;

    for result in results {
        println!("Row {}: distance {:.4}", result.row_idx, result.distance);
    }

    Ok(())
}
```

### 3) DataFusion SQL

```rust
use std::path::Path;
use std::sync::Arc;

use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use pq_vector::df_vector::{VectorTopKOptions, VectorTopKPhysicalOptimizerRule};
use pq_vector::{EmbeddingColumn, IndexBuilder, IvfBuildParams};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = Path::new("data/embeddings.parquet");
    let indexed = Path::new("data/embeddings_indexed.parquet");

    if !indexed.exists() {
        IndexBuilder::new(source, indexed, EmbeddingColumn::try_from("embedding")?)
            .params(IvfBuildParams::default())
            .build()?;
    }

    let options = VectorTopKOptions {
        nprobe: 8,
        batch_size: 1024,
        max_candidates: None,
    };
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(VectorTopKPhysicalOptimizerRule::new(options)))
        .build();
    let ctx = SessionContext::new_with_state(state);

    ctx.register_parquet("t", indexed.to_str().unwrap(), ParquetReadOptions::default())
        .await?;

    let df = ctx
        .sql("SELECT id FROM t ORDER BY array_distance(embedding, [0.0, 0.0]) LIMIT 5")
        .await?;
    let _batches = df.collect().await?;
    Ok(())
}
```

## In-place indexing (optional)

Append an IVF index to an existing Parquet file without rewriting data pages:

```rust
use pq_vector::{EmbeddingColumn, InplaceIndexBuilder, IvfBuildParams};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    InplaceIndexBuilder::new(
        Path::new("data/embeddings.parquet"),
        EmbeddingColumn::try_from("embedding")?,
    )
    .params(IvfBuildParams::default())
    .build()?;

    Ok(())
}
```

## Notes

- `k` controls how many results you return.
- `nprobe` trades speed for recall (higher = more accurate, slower).

## License

MIT
