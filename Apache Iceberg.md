---
aliases:
  - Iceberg
tags: Catalog
---
==An open table format for large analytic datasets from Netflix (2017), designed to sit on top of [[Apache Parquet|Parquet]] files in [[Blob Storage|Object Storage]] like [[Amazon S3|S3]], [[Google Cloud Storage|GCS]], etc, letting it behave more like a database table.==
- Many different query layers are adopting Iceberg as a specification.

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


![[Pasted image 20260425131748.png]]



# Comparison with tools like [[Apache Gravitino|Gravitino]] or [[Apache Polaris|Polaris]]
- ==Iceberg== is **how a table is physically organized and versioned***: A table format. "How should this table exist?"
	- An open table format for huge analytic datasets, defining things like table schema, partitions, snapshots/time travel, hidden partitioning, schema evolution ACID transactions, metadata fields + manifests, object storage layout (eg on S3)
	- Replaces fragile [[Apache Hive|Hive]]-style tables with something more reliable for modern [[Data Lakehouse]]s.
	- Used by [[Apache Spark]], [[Trino]], [[Apache Flink|Flink]], [[DuckDB]], [[Snowflake]] (interop), many others
- ==Gravitino== is **How metadata across many systems is managed and governed.**: A metadata management and federation layer. "How do I manage ALL my tables across ALL my systems?"
	- Sits above systems like Iceberg, helping to manage many catalogs, many engines, many table formats, access control, governance, lineage, discovery, unified metadata APIs
	- It can unify metadata across: [[Apache Iceberg|Iceberg]], [[Delta Lake]], [[Apache Hive|Hive]], [[Apache Hudi]], [[Java Database Connectivity|JDBC]], [[Blob Storage|Object Storage]], ML assets, files, catalogs

Iceberg: The blueprint + transaction log for one building. Helps: "This NDVI table is versioned correctly."
Gravitino: The city planning for all buildings, knowing where everything is, who can access it, and how systems relate. Helps: "Where are all the environmental tables, who owns them, and who can query them."

You don't choose one instead of the other; they're complementary, not competitors.
