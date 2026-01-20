# pq-vector

Embed IVF (Inverted File) vector indexes directly into Parquet files for fast approximate nearest neighbor search.

## Features

- **Embedded Index**: Index stored within the Parquet file itself - no separate index files
- **Standard Compatible**: Indexed files remain valid Parquet - DuckDB, Pandas, etc. can read them normally
- **DataFusion Integration**: SQL table function for vector search queries

## Quick Start

### 1. Build an Index

Convert an existing Parquet file with embeddings into an indexed file:

```rust
use pq_vector::{build_index, IvfBuildParams};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = IvfBuildParams {
        n_clusters: Some(100),  // Number of IVF clusters (default: sqrt(n))
        max_iters: 20,          // K-means iterations
        seed: 42,
    };

    build_index(
        Path::new("data/embeddings.parquet"),      // Source file
        Path::new("data/embeddings_indexed.parquet"), // Output file
        "embedding",                                // Column name containing vectors
        &params,
    )?;

    Ok(())
}
```

### 2. Search with Rust API

```rust
use pq_vector::topk;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query_vector: Vec<f32> = vec![/* your query embedding */];

    let results = topk(
        Path::new("data/embeddings_indexed.parquet"),
        &query_vector,
        10,    // k: number of results
        5,     // nprobe: clusters to search (higher = more accurate, slower)
    )
    .await?;

    for result in results {
        println!("Row {}: distance {:.4}", result.row_idx, result.distance);
    }

    Ok(())
}
```

### 3. DataFusion SQL Integration

```rust
use datafusion::prelude::*;
use pq_vector::{encode_query_vector, TopkBinaryTableFunction};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = SessionContext::new();
    
    // Register the topk_bin table function
    ctx.register_udtf("topk_bin", Arc::new(TopkBinaryTableFunction));

    // Encode your query vector as base64
    let query_vector: Vec<f32> = vec![/* your query embedding */];
    let query_b64 = encode_query_vector(&query_vector);

    // Query with SQL
    let df = ctx.sql(&format!(
        "SELECT title, _distance 
         FROM topk_bin('data/embeddings_indexed.parquet', '{}', 10, 5)
         LIMIT 5",
        query_b64
    )).await?;

    df.show().await?;
    Ok(())
}
```

**Output:**
```
+--------------------------------------------------+------------+
| title                                            | _distance  |
+--------------------------------------------------+------------+
| Most similar document                            | 0.0        |
| Second most similar                              | 0.42156    |
| Third most similar                               | 0.51823    |
+--------------------------------------------------+------------+
```

## Table Functions

### `topk_bin` (recommended for high-dimensional vectors)

```sql
SELECT * FROM topk_bin(path, base64_query, k, nprobe)
```

- `path`: Path to indexed Parquet file
- `base64_query`: Query vector encoded with `encode_query_vector()`
- `k`: Number of results to return
- `nprobe`: Number of clusters to search (default: 10)

### `topk` (for small vectors)

```sql
SELECT * FROM topk(path, ARRAY[1.0, 2.0, ...], k, nprobe)
```

Note: Use `topk_bin` for vectors with many dimensions (>100) due to SQL parsing limitations.

## Result Columns

The table functions return all original columns plus:

| Column | Type | Description |
|--------|------|-------------|
| `_distance` | Float32 | L2 distance from query vector |
| `_row_idx` | UInt32 | Original row index in Parquet file |

## How It Works

1. **Index Building**: K-means clustering creates IVF centroids, vectors are assigned to clusters
2. **Storage**: Index bytes (centroids + inverted lists) appended after Parquet data pages, offset stored in footer metadata
3. **Search**: Query finds nearest centroids, searches only those clusters, returns top-k by distance

## Parameters

### `IvfBuildParams`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clusters` | `sqrt(n)` | Number of IVF clusters |
| `max_iters` | 20 | K-means iterations |
| `seed` | 42 | Random seed for reproducibility |

### Search Parameters

| Parameter | Recommendation | Description |
|-----------|----------------|-------------|
| `k` | - | Number of results needed |
| `nprobe` | 5-20 | More = better recall, slower. Start with `sqrt(n_clusters)` |

## Recall vs Speed

With 50 clusters on 496 vectors:

| nprobe | Recall@10 | Clusters Searched |
|--------|-----------|-------------------|
| 1 | ~49% | 2% |
| 5 | ~87% | 10% |
| 10 | ~95% | 20% |
| 50 | 100% | 100% |

## Requirements

```toml
[dependencies]
pq-vector = "0.1"
datafusion = "52.0"  # For SQL integration
tokio = { version = "1", features = ["rt-multi-thread"] }
```

## License

MIT
