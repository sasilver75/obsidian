---
tags:
  - Catalog
---
An open source storage layer that brings [[ACID]] transactions, versioning, and schema management to data stored as [[Apache Parquet|Parquet]] files on [[Blob Storage|Object Storage]].

Originally developed by [[Databricks]] and later open-sourced in 2019, now a [[Linux Foundation]] project.

Core problem: 
- [[Blob Storage|Object Storage]] is cheap but fundamentally dumb; it stores objects but knows nothing about transactions or consistency.
	- If you're writing a large dataset as many Parquet files and a failure occurs halfway through, you get a partially written, corrupted table. 
- Delta lake ==adds a transaction log on top of plain Parquet files.==

To reconstruct the current state of the table, you can replay the transaction log; the log is the source of truth; the Parquet files are just data.

The log enables:
- ACID transactions
- Time travel
- Schema evolution
- Schema enforcement
- Upserts and delete


![[Pasted image 20260417175836.png]]


