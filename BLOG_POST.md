# Native Vector Search in Parquet with DataFusion

To search vector embeddings, the standard advice is almost always "spin up a vector database."

Tools like Pinecone, Milvus, or specialized formats like Lance are fantastic pieces of engineering.
But they come with a hidden cost: complexity. New infrastructure to manage, new file formats to learn, and new data to coordinate.

But what if you didn't need any of that? What if you could just keep your data in Parquet, the format you're already using, and still get fast vector search?

This post explores a little experiment: implementing efficient, native vector search directly inside Parquet files.

## Wait, isn't Parquet terrible for random access?

"But Parquet is a columnar format!" I can hear you screaming. "It's designed for heavy scans, not point lookups!"

You’re not wrong. The common wisdom is that Parquet is ill-suited for random access because it compresses data into pages.
To read a single row, you typically have to decompress an entire page, which is wasteful if you only want one item.

But let's pause and look at the actual numbers for a second.

A typical vector embedding—say, from OpenAI's `text-embedding-3-small` model—has 1,536 dimensions.
That's about 6KB of data. Now, guess how big a standard Parquet page is? usually a few KB as well.

Do you see where I'm going with this?

If we simply configure the Parquet writer to align the page size with the embedding size, we can effectively force each embedding into its own page.
With this, "decompressing a page" just means "reading the one vector we want."

We don't change the file format; we just tune it.

```rust
// Configure parquet to store one embedding per page
let props = WriterProperties::builder()
    .set_data_page_size_limit(vector_size)     // one vector per page
    .set_data_page_row_count_limit(1)          // enforce it
    .set_column_compression(embedding_col, Compression::LZ4_RAW)
    .build();
```

With this simple configuration change, we've effectively turned Parquet into a random-access friendly format for our vectors.

## Zero-Copy Vector Indexing

Of course, fast random access isn't enough. If you have to scan every single row to calculate distances (O(N)), it doesn't matter how fast you can read an individual page — it’ll still be slow. We need an index.

But here’s the challenge: how do we add an index without breaking compatibility?
We don't want to create a "custom Parquet" that DuckDB or Spark can't read.

The solution turns out to be surprisingly elegant.
Parquet allows you to embed arbitrary metadata in the file footer (more details [here](https://datafusion.apache.org/blog/2025/07/14/user-defined-parquet-indexes/)).
Standard readers will happily ignore it, but our specialized reader can look for it.

We chose the Inverted File (IVF) index for this prototype.
It works by partitioning the vector space into clusters (centroids).
When we want to search, we figure out which clusters are close to our query, and then we only look at the vectors in those clusters.

The best part? It's **zero-copy**.

Some vector stores (HNSW-based) force you to duplicate your data into their internal structures.
Our index is just a lightweight list of pointers (row IDs) and cluster centroids.
The actual heavy vector data stays right where it is — in the Parquet data pages.

In our experiments with ~5,000 academic papers each with 4096-dimensional embeddings (we used `qwen/qwen3-embedding-8b`), the index added a negligible **0.21 MB** to a **68 MB** file.
That's an overhead of just **0.3%**.

## How it looks in code

We implemented a proof-of-concept using Rust, leveraging the `parquet` and `datafusion` crates.

### 1. Building the Index

The builder reads your existing Parquet file, trains K-means centroids, and writes out a new, indexed Parquet file.

```rust
use pq_vector::{build_index, IvfBuildParams};

build_index(
    Path::new("data/combined.parquet"),
    Path::new("data/combined_indexed.parquet"),
    "embedding",  // embedding column name
    &IvfBuildParams::default()
)?;
```

### 2. Searching with SQL (DataFusion)

To make this actually useful, we hooked it into DataFusion via a User-Defined Table Function (UDTF).
This lets you run vector searches using standard SQL syntax.

```rust
let ctx = SessionContext::new();
ctx.register_udtf("topk", Arc::new(TopkTableFunction));

let query_array = query_vector
    .iter()
    .map(|v| format!("{:.6}", v))
    .collect::<Vec<_>>()
    .join(", ");

let df = ctx.sql(&format!(
    "SELECT title, _distance
     FROM topk('data/combined_indexed.parquet', ARRAY[{}], 10, 5)
     WHERE year >= '2023'",
    query_array
)).await?;
```

Notice that we can mix the vector search (`topk`) with normal SQL filters (`WHERE year >= '2023'`).
This is the power of keeping everything in one engine, you don't need a vector search engine.

## Does it actually work?

We benchmarked this on our dataset of 4,886 vectors (4096 dimensions).

| Operation              | Time   | Speedup  | Recall@10 |
| ---------------------- | ------ | -------- | --------- |
| Brute force            | 100ms  | 1x       | 1.00      |
| IVF search (nprobe=1)  | 3.4ms  | **29x**  | 0.83      |
| IVF search (nprobe=5)  | 17.7ms | **5.7x** | 0.96      |
| IVF search (nprobe=10) | 32.4ms | **3.1x** | 1.00      |

The results are pretty clear. Even with a simple IVF index, we see massive speedups compared to a full scan. With `nprobe=5`, we're getting **96% recall** at **5.7x the speed** of a brute force scan.

The latency is almost entirely dominated by random I/O, which validates our theory: if you tune the page size correctly, Parquet handles random access just fine.

## Limitations and Looking Ahead

Now, I don't want to oversell this. This is a prototype, and we're definitely trading some things for simplicity:

- **No HNSW**: We used IVF because it's simple and compact. Graph-based indexes like HNSW are probably more accurate, but they have much higher space overhead.
- **DataFusion-native index**: It would be nice to query like `SELECT * FROM table ORDER BY cosine_similarity(vector_column, vector) LIMIT 10`.
- **Multi-parquet index**: We often query many parquet files, not just one file. Making our system designed for multi-index would be great.

[github.com/XiangpengHao/pq-vector](https://github.com/XiangpengHao/pq-vector)
