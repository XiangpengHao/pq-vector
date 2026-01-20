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

The vector index will be inserted as a custom index in the Parquet file, more can be found here: https://datafusion.apache.org/blog/2025/07/14/user-defined-parquet-indexes/

But if we change the column data, how can users still read their vector data?

- So we don't. We use ivf, which is just pointers pointing to the actual data.

### How to leverage the index?

We should create a user-defined-table-function in DataFusion.

```sql
SELECT *
FROM topk('/path/to/file.parquet', 'col_embedding', :query_vector, 100)
WHERE category = 'systems' LIMIT 32;
```

But it seems that DataFusion doesn't support arbitrary table functions yet.
https://github.com/apache/datafusion/issues/7926
https://github.com/apache/datafusion/issues/8383

Need to double check.

### Testing

1. First we will rewrite a parquet file. One example parquet file is in data/vldb_2025.parquet.
2. Then we will use the aforementioned topk style to query the file.
3. Finally we show that it works.
