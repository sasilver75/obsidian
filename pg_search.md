A Rust-based [[PostgreSQL|Postgres]] extension that significantly improves Postgres's [[Full-Text Search Index|Full-Text Search]] capabilities.

Today, Postgres's native full text search has two main problems:
1. Performance: Searching and ranking over large tables is sluggish when tables grow to millions of rows; a ***single full-text search can take several minutes.***
2. Functionality: Postgres has no support for operations like ==fuzzy search==, ==relevance tuning==, or ==[[BM25]] relevance scoring==, which are the bread and butter of modern search engines.

==`pg_search` aims to bridge the gap between the native capabilities of Postgres's full text search, and those of a specialized search engine like [[ElasticSearch]].==
- The goal is to *eliminate the need* to bring a cumbersome service like ElasticSearch into the data stack.

Some features of `pg_search` include:
- 100% Postgres native, with zero dependencies on an external search engine.
- Built on top of [[Tantivy]], a Rust-based alternative to the [[Apache Lucene]] search library.
	- 2016: `Tantivy` designed as a Rust-based alternative to Lucene
		- Uses an [[Inverted Index]], which stores a mapping from words to their locations in a set of documents. `pg_search` stores the index inside Postgres as a new, Postgres-native index type, which we call the BM25 index.
		- When a BM25 index is created, Postgres automatically updates it as new data arrives or is deleted in the underlying SQL table; it enables real-time search without additional re-indexing logic.
	- 2019: `pgrx` library made it possible to build Postgres extensions in Rust.
- Query times over 1M rows are ==20x faster compared to `tsquery` and `ts_rank`, Postgres' built-in full text search and sort functions!==
- Support for ==fuzzy search==, aggregations, highlighting, and ==relevance tuning==.
- Relevance scoring uses [[BM25]], the same algorithm used by ElasticSearch.
- Real-time search: ==New data is immediately searchable without manual reindexing==.



