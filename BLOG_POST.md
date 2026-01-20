How to vector search with Parquet and DataFusion?

## What is vector search? why it's important?

## What are other approaches?

Notably, lance.

#### Drawbacks

- Need to do a entirely new file format (lance).
- Need a specialized query engine (i.e., lancedb)

## Why Parquet and DataFusion?

We like Apache Parquet and Apache DataFusion.

- Open source
- Open governance
- Open contribution

## Wait, is it slow/impossible?

### Is it possible?

Parquet is a general-purpose columnar format, does it support vector search?

Well, vector embeddings are just number arrays, and any serious file format supports number arrays.

### Is it fast?

You may think, humm, but vector search is not a first-class citizen in Parquet, how can it be fast?

Indeed, If you look at lance's argument (cite to lance's blog post), Parquet is slow/unsuitable for vector search, particularly because it doesn't allow random lookups to individual embeddings.

But that is only part of the story. Parquet is for sure compressed at page granularity, and each page is a few KB.
But wait, how large is a vector embedding? The most popular embedding model (according to [OpenRouter](https://openrouter.ai/models?fmt=cards&output_modalities=embeddings&order=most-popular) is OpenAI's `text-embedding-3-small`) is 1,536 dimensions, each value is 4 bytes, so a vector embedding is 6KB, a perfect size for just one page in Parquet.
This means that with proper Parquet configuration, we can easily decompress each individual embedding, no extra work.

This shows the power of Parquet: its rich configuration space allows us to optimize for various kinds of workloads, including vector embeddings. (one of my never-finishing side project is to automatically optimize Parquet files by tuning Parquet's configurations).

## What about vector indexes?

Now you're convinced Parquet can be efficient to **store** embeddings, but one big feature of LanceDB is to quickly **search** embeddings. How can you even do this, Parquet is born before LLM is a thing!

Some smart folks earlier found that _You can embed anything in Parquet without breaking the format_, this means we can comfortably embed the vector index in the Parquet file -- if the reader knows how to use it, it can use it; if the reader doesn't recongized it, it just ignore it, won't break anything!

Here's our approach:

1. Read the embedding column. (show a code example here)
2. Build a vector index and embed it in the SAME parquet file as rest of the data. (show a size comparison of with and without vector index)
3. Use the vector index to search embeddings. (show a programmtical example use of it)

## How can I query it in DataFusion?

Having demonstrated that using our library, you can programmatically query the top k using the vector index. But how can I query it in DataFusion?

(show the datafusion integration here)

## Future work

This is definitely not perfect, but maybe a good start.

Here are a few things we can improve:

- Truly support DataFusion vector index. Right now it's sort of a hack: we always needs to read top k first, then apply other database operations.
  A truly DB-native approach is to allow writing queries like `SELECT * from table ORDER BY cosine_similarity(embedding, vector) LIMIT 10`.

- We only implemented IVF, a better approach is to implement IVF-pq, just like lance does. This can significantly reduce the memory footprint of the index.

- We only implemented cosine similarity, other distance metrics can also be added.

- We only support index-per-parquet-file, but it would be great to have a good story for multiple parquet files support. How to coordinate the index across multiple files?
