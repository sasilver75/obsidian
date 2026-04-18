The most common data organization pattern within a [[Data Lakehouse]].

Three layers of data quality:
1. ==Bronze== (raw): Data exactly as it arrived; no transformations, append-only. Preserves full history, and if anything goes wrong, you can reprocess from bronze.
2. ==Silver== (cleaned): Deduplicated, validated, standardized. Bad records filtered or corrected. Joined across sources. Still relatively raw, but trustworthy.
3. ==Gold== (curated): Business-level aggregations, metrics, features. Optimized for specific use cases (BI dashboards, ML feature stores, reporting). What most analysts and BI tools query.

Each layer is typically a set of [[Delta Lake]]/[[Apache Iceberg]] tables in [[Blob Storage|Object Storage]].

Transformation between layres is typically done with [[Apache Spark|Spark]] or [[dbt]].




