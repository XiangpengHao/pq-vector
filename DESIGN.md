pq-vector: Parquet with vector index

Two core features:

1. Float number compression
2. Vector index

## Float number compression

### ALP

I initially though ALP (https://dl.acm.org/doi/10.1145/3626717) will help.
But ALP only compresses well for floating numbers that are initially decimal numbers.

For true floating numbers like vector embeddings, ALP won't help.

### General purpose compression

We can still use general purpose compression algorithms like LZ4 or ZSTD.
One complaint is that Parquet needs to de/compress entire page, i.e., no random access.

One vector is 1024\*4 = 4KB, we can easily just set the page size to 4KB.
This way we can do random access.

## Vector index

We can do hnsw, but hnsw is very wasteful, the index is larger than the data.

We can just do ivf, with optional pq.
The nice thing about ivf is that for each list, we can just point to the actual data (file offset), so very compact.
Remember that each of the page is just one vector, so random-access friendly.
We follow the standard practice to choose k = sqrt(n).

## Ecosystem integration

We keep the vector index embedded in the Parquet file without altering the column data, so existing readers can still read the original vectors.
