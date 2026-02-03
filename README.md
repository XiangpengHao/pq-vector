# pq-vector

Embed IVF (Inverted File) vector indexes directly into Parquet files for fast approximate nearest neighbor search.

## Features

- **Embedded Index**: Index stored within the Parquet file itself - no separate index files
- **Standard Compatible**: Indexed files remain valid Parquet - DuckDB, Pandas, etc. can read them normally
- **DataFusion integration**: Ergonomic vector search with just SQL.
- **Zero-copy**: Zero-copy, in-place Parquet indexing.

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

### 1b. Build an Index In-Place

Append an IVF index to an existing Parquet file without rewriting the data pages:

```rust
use pq_vector::{build_index_inplace, IvfBuildParams};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = IvfBuildParams::default();

    build_index_inplace(
        Path::new("data/embeddings.parquet"),
        "embedding",
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

## Result Columns

The Rust API returns row indices and distances:

| Column     | Type  | Description                   |
| ---------- | ----- | ----------------------------- |
| `row_idx`  | usize | Row index in the Parquet file |
| `distance` | f32   | L2 distance from query vector |

## How It Works

1. **Index Building**: K-means clustering creates IVF centroids, vectors are assigned to clusters
2. **Storage**: Index bytes (centroids + inverted lists) appended after Parquet data pages, offset stored in footer metadata
3. **Search**: Query finds nearest centroids, searches only those clusters, returns top-k by distance

## Parameters

### `IvfBuildParams`

| Parameter    | Default   | Description                     |
| ------------ | --------- | ------------------------------- |
| `n_clusters` | `sqrt(n)` | Number of IVF clusters          |
| `max_iters`  | 20        | K-means iterations              |
| `seed`       | 42        | Random seed for reproducibility |

### Search Parameters

| Parameter | Recommendation | Description                                                 |
| --------- | -------------- | ----------------------------------------------------------- |
| `k`       | -              | Number of results needed                                    |
| `nprobe`  | 5-20           | More = better recall, slower. Start with `sqrt(n_clusters)` |

## Recall vs Speed

With 50 clusters on 496 vectors:

| nprobe | Recall@10 | Clusters Searched |
| ------ | --------- | ----------------- |
| 1      | ~49%      | 2%                |
| 5      | ~87%      | 10%               |
| 10     | ~95%      | 20%               |
| 50     | 100%      | 100%              |

## Requirements

```toml
[dependencies]
pq-vector = "0.1"
tokio = { version = "1", features = ["rt-multi-thread"] }
```

## License

MIT
