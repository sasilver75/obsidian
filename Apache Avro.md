---
aliases:
  - Avro
---


Initially released in 2009, created by Doug Cutting to be an open-source, flexible, and efficient ==data serialization system==.
- Avro is a ==row-based== ==serialization format==, storing all data for a single record together.
- Makes it ==ideal for write-heavy operations==, like streaming data, data ingestion, rather than analytical queries.
	- [[Kafka]] uses it heavily; think of it as "==Fast, compact JSON with a schema==."

Key aspects:
- Schema-driven: Avro data is always stored with its schema in JSON format, allowing for full data description without explicit type tagging.
- Binary encoding: Primarily uses a compact binary format for efficient transmission and storage, though JSON is also supported.
- Schema Evolution: Ensures backwards and forwards compatability, which is essential for data persistence in systems like Apache Kafka.


