---
aliases:
  - Polaris
---
An open-source ==catalog and governance layer for modern lakehouses==, designed to provide a unified way to manage tables, metadata, and access policies across multiple engines and storage systems.
- It acts as the "control plane" for data assets, especially for open table formats like [[Apache Iceberg|Iceberg]], so that engines such as [[Apache Spark|Spark]], [[Trino]], [[Apache Flink|Flink]] can all work against the same consistent catalog, instead of maintaining separate metadata silos.
- It helps organizations avoid vendor lock-in by using open standards rather than proprietary warehouse catalogs.
- It manages:
	- namespaces
	- schemas
	- table definition
	- permissions
	- discovery
- Instead of each compute engine having its own view of where data lives and who can access it, Polaris centralizes that logic, supporting fine-grained governance such as [[Role-Based Access Control]] (RBAC), auditing, and policy enforcement, while exposing standard APIs so different systems can interoperate cleanly.
	- A geospatial analytics team might store massive satellite features tables in [[Apache Iceberg|Iceberg]] on [[Blob Storage|Object Storage]] while analysts query them from multiple engines; Polaris ensures everyone is referencing the same metadata and governance rlules.

![[Pasted image 20260425131956.png]]




# Apache Polaris vs [[Apache Gravitino]]
- Both trying to solve metadata management in modern lakehouses, but they focus at different layers with slightly different philosophies.
	- ==Polaris== is primarily a catalog for open data formats, especially centered around [[Apache Iceberg]]. Its main goal is to provide a unified, open control plane for tables so multiple engines ([[Apache Spark|Spark]], [[Trino]], [[Apache Flink|Flink]]) can share the same namespaces, permissions, table definitions. "One catalog for all my lakehouse tables."
	- ==Gravitino== is broader: A metadata lakehouse, rather than a table catalog. It aims to unify metadata not only for [[Apache Iceberg|Iceberg]] tables, but also for [[Delta Lake]], [[Apache Hudi]], [[Apache Hive|Hive]], [[Java Database Connectivity|JDBC]] databases, files, object storage, ml assets, and more.
- Analogy
	- Polaris is like the official registry for your buildings, while Gravitino is the city planning office for the whole city.
		- Polaris is more focused, tighter around open-table lakehouse operations, and especially strong if your architecture is Iceberg-first.
		- Gravitino is more expansive and better suited when your organization has many engines, many storage models, and many metadata domains beyond tables.