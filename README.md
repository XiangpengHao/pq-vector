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
use pq_vector::IndexBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    IndexBuilder::new(
        "data/embeddings.parquet", // Source file (indexed in-place by default)
        "embedding",               // Column name containing vectors
    )
    .n_clusters(100)
    .max_iters(20)
    .seed(42)
    .build_inplace()?;

    // Optional: write to a new file instead of in-place
    IndexBuilder::new("data/embeddings.parquet", "embedding")
        .build_new("data/embeddings_indexed.parquet")?;

    Ok(())
}
```

### 2) Search with Rust

```rust
use pq_vector::TopkBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query_vector: Vec<f32> = vec![/* your query embedding */];

    let results = TopkBuilder::new("data/embeddings_indexed.parquet", &query_vector)
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
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use pq_vector::df_vector::{PqVectorSessionBuilderExt, VectorTopKOptions};
use pq_vector::IndexBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let indexed = "data/embeddings_indexed.parquet";

    let options = VectorTopKOptions {
        nprobe: 8,
        max_candidates: None,
    };
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_pq_vector(options) // ENABLE pq-vector here!
        .build();
    let ctx = SessionContext::new_with_state(state);

    ctx.register_parquet("t", indexed, ParquetReadOptions::default())
        .await?;

    let df = ctx
        .sql("SELECT id FROM t ORDER BY array_distance(embedding, [0.0, 0.0]) LIMIT 5")
        .await?;
    let _batches = df.collect().await?;
    Ok(())
}
```

If you already have a custom `SessionConfig`, enable pq-vector like this:

```rust
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionConfig;
use pq_vector::df_vector::{PqVectorSessionBuilderExt, PqVectorSessionConfigExt, VectorTopKOptions};

let options = VectorTopKOptions {
    nprobe: 8,
    max_candidates: None,
};
let config = SessionConfig::new()
    .with_target_partitions(2)
    .with_pq_vector();
let state = SessionStateBuilder::new()
    .with_default_features()
    .with_pq_vector(options)
    .with_config(config)
    .build();
```

## Notes

- `k` controls how many results you return.
- `nprobe` trades speed for recall (higher = more accurate, slower).

## License

MIT
