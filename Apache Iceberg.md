---
aliases:
  - Iceberg
---
==An open table format for large analytic datasets from Netflix (2017), designed to sit on top of [[Blob Storage|Object Storage]] like [[Amazon S3|S3]], [[Google Cloud Storage|GCS]], etc, letting it behave more like a database table.==

Core problem it solves:
- You have millions of [[Apache Parquet|Parquet]] files in S3, and querying them is painful:
	- You have to scan everything
	- You can't do atomic updates
	- Multiple writers step on eachother

Iceberg adds:
- Schema evolution: Add/rename/drop columns without rewriting all your files
- Time travel: Query the table as it looked at any past snapshot
- [[ACID]] transactions. Atomic commits, concurrent writers don't corrupt eachother.
- Partition evolution: Change how data is partitioned without rewriting history
- Hidden partitioning: The query engine figures out which files to skip automatically.

How it works conceptually:
- Iceberg maintains a ==metadata layer==: A ==catalog== and ==manifest== files, which track exactly which data files make up the current (and historical) table state.
- The actual data is still in [[Apache Parquet|Parquet]]/[[Optimized Row Columnar|ORC]]/[[Apache Avro]]. ==Iceberg is the book-keeping layer on top.==

It's used in [[Data Lake]]s at scale @ Netflix, Apple, LinkedIn
- Typically paired with query engines like [[Apache Spark|Spark]], [[Trino]], [[DuckDB]]
- Shines when you have PB of data across thousands of files with multiple concurrent writers.
- You'd reach for Iceberg if you were building a data warehouse layer separate from the operaitonal database, like archiving all raw ingest data to [[Cloudflare R2]]