---
tags:
  - Catalog
---
A very new open [[Data Lakehouse]] catalog format announced by [[DuckDB Labs]] in early 2025, designed to be a dramatically simpler alternative to existing lakehouse catalog solutions.

Claim: Modern data lakehouses (built on [[Apache Iceberg]], [[Delta Lake]]) need a ==catalog==: A metadata store that tracks what tables exist, where their data files live, what schema they have, what snapshots exist for time travel, etc.

Existing catalog options are heavy:
- Apache Hive Metastore: Requires running service + Relational DB backend
- [[AWS Glue]]: Managed but AWS-only
- [[Apache Polaris]], [[Apache Gravitino]]: Newer, still require running services
- Project Nessie: Git-like Catalog, still a running service

All of these require infrastructure to operate.

DuckLake's approach:
- Use a [[DuckDB]] database file as the catalog.
- That's it.
- ==The entire catalog (all table metadata, snapshots, schema history, transaction log, etc.) lives inside a *single DuckDB file* that can sit on S3, GCS, or local disk.==
```
  s3://mybucket/catalog.ducklake    ← the entire catalog
  s3://mybucket/data/               ← the actual Parquet data files
```

Key properties:
- Serverless (no catalog service to run; just a file)
- ACID transactions (inherited from [[DuckDB]])
- Time travel (Like [[Apache Iceberg|Iceberg]], DuckLake tracks snapshots so you can query historical versions of tables)
- Iceberg compatibility (Can export/expose tables as Iceberg metadata, so existing Iceberg-compatible tools ([[Apache Spark|Spark]], [[Trino]], [[Apache Flink|Flink]]) can read DuckLake tables)
- Format agnostic: Data files are [[Apache Parquet|Parquet]] or other formats; the catalog is separate from the data.


![[Pasted image 20260417175530.png]]
