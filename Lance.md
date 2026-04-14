"The Open [[Data Lakehouse|Lakehouse]] Format for Multimodal AI"

A ==columnar storage format== designed for ML datasets (images, embeddings, geometries together). Think [[Apache Parquet|Parquet]] but with native support for tensors and vector [[Embedding]]s. Relevant if you're building GeoAI pipelines where each row has both geometry and embedding vectors.

Contains:
- A file format
- Table format
- Catalog specification
That allows you to build a complete lakehouse on top of [[Blob Storage|Object Storage]] to power your AI workflows.

Useful for:
- Building search engines and feature stores with hybrid search capabilities
- Large-scale ML training requiring high performance IO and random access
- Storing, querying, and managing multimodal data including images, videos, audio, text, and embeddings

Compatible with [[Pandas]], [[DuckDB]], [[Polars]], [[Apache Arrow|PyArrow]], [[Ray]], [[Spark]], and more.

Key Features:
- Expressive [[Hybrid Search]]: Combines vector similarity search, full-text search ([[BM25]]), and SQL analytics on the same dataset with accelerated secondary index.
- Lightning-fast random access: 100x faster than [[Apache Parquet|Parquet]] or [[Apache Iceberg|Iceberg]] for random access without sacrificing scan performance.
- Native multimodal data support: Store images, videos, audio, text, and embeddings in a single unified format with efficient blob encoding and lazy loading.
- Data evolution: Efficiently add columns with backfilled values without full table rewrites, perfect for ML feature engineering.
- Zero-copy versioning: Automatic versioning with [[ACID]] transactions, time travel, tags, and branches, with no infrastructure needed.
- Rich ecosystem integration: [[Apache Arrow|Arrow]], [[Pandas]], [[Polars]], [[DuckDB]], [[Spark]], [[Ray]], [[Trino]], [[Apache Flink|Flink]], and open catalogs (e.g. [[Apache Polaris|Polaris]], [[Databricks Unity Catalog|Unity Catalog]], [[Apache Gravitino|Gravitino]])
