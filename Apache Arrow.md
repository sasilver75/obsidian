---
aliases:
  - PyArrow
  - Arrow
---
---


An ==in-memory columnar tabular data format==.
- A specification for how to lay out tabular data in RAM so that ==different tools can share it without copying or serializing==. Think of it as a ==common language for data in memory==.
	- If [[Pandas]], [[DuckDB]], and some custom C++ tool all agree to use Arrow's memory layout, they can pass data between eachother as a pointer (zero copy, zero serialization). This is the "==zero-copy IPC (inter-process communication==" that makes modern data pipelines fast!

==Arrow is NOT a file format, it is a MEMORY FORMAT.==
- (There's an Arrow IPC file format for streaming/saving Arrow data, but that's secondary)


## Relation to [[Apache Parquet|Parquet]]
- They're complementary, not competing:
```
    ┌─────────────┬────────────────────┬─────────────────────┐
    │             │       Arrow        │       Parquet       │
    ├─────────────┼────────────────────┼─────────────────────┤
    │ Where       │ In memory          │ On disk             │
    ├─────────────┼────────────────────┼─────────────────────┤
    │ Purpose     │ Fast computation   │ Efficient storage   │
    ├─────────────┼────────────────────┼─────────────────────┤
    │ Compression │ None (fast access) │ Heavy (small files) │
    ├─────────────┼────────────────────┼─────────────────────┤
    │ Columns     │ Yes                │ Yes                 │
    └─────────────┴────────────────────┴─────────────────────┘
```
- Above: For Parquet, think "Park it here", in terms of on-disk storage, whereas Arrows are fast, flying through the air, and are for that fast IPC.
- Typical flow: 
	- Read Parquet data from disk
	- Decompress into Arrow in memory
	- Query/transform with DuckDB or Pandas
	- Write back to Parquet
- ==Parquet is the storage format, Arrow is the compute format.==


## GeoArrow
- [[GeoArrow]] is to Arrow what [[GeoParquet]] is to Parquet -- just a specification for encoding geometry in Arrow's columnar format.
- Geometry operations can be vectorized over Arrow arrays using [[SIMD]], which is dramatically faster than deserializing [[Well-Known Binary|WKB]]s row-by-row.



