# Vector Search with Parquet and DataFusion

You've got embeddings. Millions of them. Now what?

The obvious answer is "spin up a vector database." But what if I told you that boring old Parquet files can do vector search too? No new file format. No specialized database. Just Parquet, the format that's been quietly storing your data since 2013 while you chased shinier things.

## The Problem with Vector Databases

Don't get me wrong—vector databases are great. LanceDB, in particular, has done impressive work. But they come with baggage:

1. **New file format** - Lance created a whole new columnar format for vectors
2. **New query engine** - You need LanceDB to read Lance files
3. **Ecosystem lock-in** - Your data analytics tools don't speak Lance

What if you already have a parquet-based data lake? What if you're already using DataFusion, DuckDB, or Spark? Do you really want to maintain a separate vector database just for embeddings?

## Wait, Can Parquet Even Do This?

"But Parquet wasn't designed for vector search!" I hear you say.

True. But Parquet was designed to be _flexible_. Let's think about this:

**Can Parquet store embeddings?** Obviously yes. Embeddings are just arrays of floats, and Parquet handles arrays just fine.

**Can Parquet enable fast random access to individual embeddings?** Here's where it gets interesting.

The Lance team [argues](https://lancedb.com/blog/the-case-for-random-access-i-o) that Parquet is unsuitable for vector search because it compresses data at the page level. You can't just grab one row—you have to decompress an entire page.

But wait. How big is a page? A few KB. How big is a typical embedding? OpenAI's `text-embedding-3-small` (the [most popular model](https://openrouter.ai/models?fmt=cards&output_modalities=embeddings&order=most-popular)) produces 1,536 dimensions × 4 bytes = **6KB**.

That's... exactly one page.

With the right Parquet configuration, each embedding lives in its own page. Random access: solved. No extra work needed—just smart configuration.

```rust
// Configure parquet to store one embedding per page
let props = WriterProperties::builder()
    .set_data_page_size_limit(vector_size)     // one vector per page
    .set_data_page_row_count_limit(1)          // enforce it
    .set_column_compression(embedding_col, Compression::LZ4_RAW)
    .build();
```

## But What About Vector Indexes?

Storing embeddings efficiently is one thing. Searching them fast is another.

The naive approach—compute distance to every vector—is O(n). That's fine for a few thousand vectors, but not for millions.

Real vector databases use indexes like IVF (Inverted File Index), HNSW, or various tree structures. Can we do this with Parquet?

Yes! Here's the trick: **Parquet lets you embed arbitrary data in the file without breaking the format.** Standard parquet readers will ignore it; specialized readers can use it.
More details read [here](https://datafusion.apache.org/blog/2025/07/14/user-defined-parquet-indexes/).

Our approach:

1. Build an IVF index on the embedding column
2. Serialize the index and append it after the data pages
3. Store the index offset in parquet's key-value metadata

Any parquet reader can still read the file normally. Our specialized reader also loads the index for fast search.

### Even better: zero-copy vector indexes

We choose IVF index because it allows a even fancier feature: zero-copy vector indexes.
The index stores pointer to the actual embeddings in the file, does not have to make a copy of the embeddings.
This makes the index much smaller and faster to load.

This is not feasible with HNSW because every embedding must be part of the index, therefore the index-to-data ratio is super high.

## Show Me The Code

Let's make this concrete. We have a parquet file with ~5,000 academic papers and their 4096-dimensional embeddings (`qwen/qwen3-embedding-8b`):

```
data/combined.parquet
├── 4,886 rows
├── 64.65 MB
└── Columns: title, authors, abstract, embedding (4096-dim)
```

### Step 1: Build the Index

```rust
use pq_vector::{build_index, IvfBuildParams};

build_index(
    Path::new("data/combined.parquet"),
    Path::new("data/combined_indexed.parquet"),
    "embedding",  // embedding column name
    &IvfBuildParams::default()
)?;
```

This creates a new parquet file with an embedded IVF index. The overhead?

| File           | Size          |
| -------------- | ------------- |
| Original       | 64.65 MB      |
| With index     | 68.21 MB      |
| Index overhead | 3.56 MB (5%)  |

The index adds just 5% to the file size—the LZ4 compression on embeddings actually saves more space than the index costs.

### Step 2: Search

```rust
use pq_vector::topk;

let results = topk(
    Path::new("data/combined_indexed.parquet"),
    &query_vector,  // your 4096-dim query
    10,             // k results
    5               // nprobe (clusters to search)
)?;

for r in results {
    println!("Row {}: distance {:.4}", r.row_idx, r.distance);
}
```

Output:

```
Row 42: distance 0.0000    <- exact match (we queried with row 42's embedding)
Row 143: distance 0.6667
Row 96: distance 0.6832
Row 187: distance 0.6839
Row 3159: distance 0.6936
```

## DataFusion Integration

The whole point of using Parquet is to stay in the ecosystem. Here's how to use vector search directly in SQL:

```rust
use datafusion::prelude::*;
use pq_vector::{TopkBinaryTableFunction, encode_query_vector};

let ctx = SessionContext::new();
ctx.register_udtf("topk_bin", Arc::new(TopkBinaryTableFunction));

// Encode the query vector as base64 (DataFusion has a [coercion bug](https://github.com/apache/datafusion/issues/19914))
let query_b64 = encode_query_vector(&query_vector);

let df = ctx.sql(&format!(
    "SELECT title, _distance
     FROM topk_bin('data/combined_indexed.parquet', '{}', 10, 5)
     WHERE year >= '2023'",
    query_b64
)).await?;
```

The `topk_bin` table function returns a table with all original columns plus `_distance` and `_row_idx`. You can filter, join, aggregate—whatever SQL can do.

## Performance

Real numbers on 4,886 vectors with 4096 dimensions:

| Operation              | Time   | Speedup    |
| ---------------------- | ------ | ---------- |
| Index build            | ~28s   | (one-time) |
| Brute force search     | ~96ms  | 1x         |
| IVF search (nprobe=1)  | ~4ms   | **24x**    |
| IVF search (nprobe=5)  | ~22ms  | **4x**     |
| IVF search (nprobe=10) | ~39ms  | **2.5x**   |

The trick? Async I/O with page-level skipping. We configured the file so each embedding lives in its own page with LZ4 compression, then use Parquet's offset index to read and decompress only the pages we need. The async reader fetches just the byte ranges for selected pages—no wasted I/O.

With nprobe=1 (searching ~1.4% of clusters), we get 24x speedup. The gap grows with dataset size as the fraction of data read shrinks.

## The Trade-offs (Honest Assessment)

This isn't a silver bullet. Here's what you're giving up:

**vs. LanceDB/specialized vector DBs:**

- No HNSW (our IVF is simpler, less accurate at high recall)
- No product quantization (higher memory usage)
- No incremental index updates (rebuild required)

But nothing fundamental, need your help to make it better!

**What you're gaining:**

- Zero new dependencies (it's just Parquet)
- Full ecosystem compatibility (DuckDB, Spark, Polars all read your files)
- One less database to operate
- Your data stays in your data lake

## Future Work

This is a proof-of-concept, not a production system. Things we'd want to add:

1. **IVF-PQ** - Product quantization to reduce memory footprint
2. **Index caching** - Keep the index in memory across queries
3. **Cosine similarity** - Currently L2 distance only
4. **Multi-file support** - Federated index across partitioned datasets
5. **Better DataFusion integration** - Make DataFusion has first-party index support.

## Try It

```bash
cargo add pq-vector

# Build an index
pq-vector index input.parquet output.parquet --column embedding

# Query from DataFusion
# ... see examples/datafusion_sql.rs
```

The code is at [github.com/XiangpengHao/pq-vector](https://github.com/XiangpengHao/pq-vector). PRs welcome.

---

_The best file format is the one you're already using._
