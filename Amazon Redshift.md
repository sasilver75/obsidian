---
aliases:
  - Redshift
---
AWS's fully managed cloud [[Data Warehouse]], a petabyte-scale relational database optimized for analytical queries ([[Online Analytical Processing|OLAP]]) rather than transactional workloads ([[Online Transactional Processing|OLTP]]).
- It's one of the dominant data warehouse platforms alongside [[Snowflake]], [[Google BigQuery|BigQuery]], [[Databricks]].
- Its main advantage is deep AWS integration: [[Amazon Identity and Access Management|IAM]], [[Amazon S3|S3]], [[AWS Glue|Glue]], [[Amazon Kinesis|Kinesis]] all work natively. If you're already heavily on AWS, Redshift is the natural choice. If you're multi-cloud, [[Snowflake]] or [[Google BigQuery|BigQuery]] are often preferred.

It achieves fast aggregations, joins, window functions over billions of rows using:
- Columnar Storage
- [[Massively Parallel Processing]] (MPP)
	- Leader nodes parse queries, build execution plans, and distribute work to compute nodes. Each compute node processes its slice of data in parallel and returns results to the leader for final aggregation.
- Query optimization specifically tuned for analytical workloads

How data is distributed across nodes matters enormously for query performance; Redshift offers (AUTO, EVEN, KEY, and ALL) distribution styles. Choosing the wrong ==distribution key== is a common source of poor query performance.

Redshift stores data on disk in sorted order by the ==sort key==. Can have ==compound sort keys== (multiple columns) and ==interleavesd sort key== (equal weight to all columns).

==Redshift Spectrum== is an important extension that lets Redshift query data directly in [[Amazon S3|S3]] ([[Apache Parquet|Parquet]], [[Optimized Row Columnar|ORC]], [[Comma-Separated Values|CSV]], [[JSON]]) without loading it into Redshift tables.
- This enables a [[Data Lakehouse]] pattern.

==Redshift Serverless== Is a newer deployment mode with no cluster management; You specify a capacity in RPUs (Redshift Processing Units), and Redshift scales automatically; you pay per second of compute used.
- Better for variable or unpredictable workloads.

Loading Data
- Redshift is optimized for bulk loads, not individual row inserts.








