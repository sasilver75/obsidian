---
aliases:
  - Gravitino
---
An open-source ==unified metadata lake==: A single layer that ==manages metadata across heterogenous data sources== and compute engines.

RATHER than having separate catalogs for each system, Gravitino federates them all into one consistent metadata layer.

It's kind of the open-source, vendor-neutral alternative to what [[Databricks Unity Catalog]] does: A single metadata control plane over a complex, multi-engine data platform.

Key capabilities:
- Multi-engine support: Works with [[Apache Spark|Spark]], [[Trino]], [[Apache Flink|Flink]], [[Apache Hive|Hive]]
- Multi-source: Spans relational DB,s data lakes, file systems, [[Apache Iceberg|Iceberg]] tables
- Centralized governance: Unified access control and permissions
- Federated queries: Query across sources through a consistent interface
