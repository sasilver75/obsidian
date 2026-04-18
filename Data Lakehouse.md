---
aliases:
  - Lakehouse
---
An architectural pattern that combines the low-cost, flexible storage of a [[Data Lake]] with the structure, [[ACID]] transactions, and query performance of a data warehouse, in a single system.
- The most common data organization pattern is [[Medallion Architecture]]

# Problem It Solves
The evolution of data architecture went through two phases, each with limitations:
- Data Warehouse (e.g. [[Amazon Redshift|Redshift]], [[Snowflake]]): Structured, schema-enforced, fast queries, ACID. Expensive storage, proprietary formats, poor support for unstructured data. Data must be loaded in, creating copies and latency.
- Data Lake (raw files on [[Amazon S3|S3]]): Cheap storage, any format, great for ML and data science. No transactions, no schema enforcement, poor query performance, data quality problems, "data swamp" syndrome. Data lands but is not organized or trusted.

... So organizations ended up building both, a data lake for raw storage and ML, and a warehouse for analytics and BI.
The [[Data Lakehouse|Lakehouse]] pattern collapses these two into one:
```
  Traditional:
    Raw data → Data Lake (S3) → ETL → Data Warehouse → BI/Analytics              
                             ↘ ML pipelines (read from lake)                                        
  Lakehouse:                                                                     
    Raw data → Lakehouse (S3 + table format) → BI/Analytics                      
                                             → ML pipelines                      
                                             → Data science
```

# How it works
- The lakehouse is enabled by ==open table formats==, a metadata and transaction layer that sits on top of regular [[Apache Parquet|Parquet]] files in [[Common Objects in Context]] and gives them warehouse-like properties.
	- [[Delta Lake]], [[Apache Iceberg]], and [[Apache Hudi]] all implement this pattern.

Cheap Object Storage + Parquet files (open, columnar, efficient) + Transaction Log/Metadata layer = ACID transactions + time travel + schema evolution + query optimization + multiple engine access

The data stays in open format files on cheap object storage. The table format adds the intelligence on top, tracking which files belong to which table version, enabling atomic writes, providing statistics for qurey optimization.

![[Pasted image 20260418003329.png]]


Major implementations:
- [[Databricks Lakehouse]] (Databricks coined the term and built [[Delta Lake]]; [[Databricks Unity Catalog|Unity Catalog]] for governance. Databricks Runtime for Spark. The most complete commercial lakehouse platform.)
- [[Apache Iceberg]] + Open Stack: Iceberg as the table format, [[Trino]] or [[Apache Spark|Spark]] for compute, and [[Apache Polaris|Polaris]] or [[Apache Gravitino|Gravitino]] as catalog. Fully open-source, multi-cloud, no vendor lock-in.
- [[Amazon Lake Formation|AWS Lake Formation]]: AWS's managed lakehouse service, integrates [[Amazon S3|S3]] + [[AWS Glue|Glue]] + [[Amazon Athena|Athena]] + [[Amazon Elastic Map Reduce|EMR]]. Less opinionated than Databricks.
- [[Google BigLake]]: [[Google BigQuery|BigQuery]] extended to read from [[Google Cloud Storage|GCS]] in open formats. Blurs the line between lakehouse and warehouse.
- [[Snowflake]]: Added [[Apache Iceberg|Iceberg]] table support, moving towards the lakehouse from the warehouse direction.


# Relevance to Geospatial
- [[GeoParquet]] as the file format (open, columnar, spatially indexed)
- [[Apache Iceberg|Iceberg]] or Delta as the table format (ACID transactions over GeoParquet files)
- [[DuckDB]] spatial or [[Apache Sedona]] as the query engine (spatial SQL over the lakehouse)
- [[SpatioTemporal Asset Catalog|STAC]] as the catalog for raster/imagery assets alongside the vector/tabular lakehouse

Organizations processing large volumes of satellite-derived [[Vector]] data (building footprints, land cover changes, environmental monitoring) are building geospatial lakehouses where imagery lives as [[Cloud-Optimized GeoTIFF|COG]]s in [[Amazon S3|S3]] and derived analytics live as [[GeoParquet]] tables managed by Iceberg.