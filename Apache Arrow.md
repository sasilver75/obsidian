---
aliases:
  - PyArrow
  - Arrow
  - Arrow Flight
  - Arrow IPC
---
An ==in-memory columnar tabular data format==, essentially a spec. for how to lay out tabular data in RAM so that different tools can ==share it without copying or serializing==.
	- If [[Pandas]], [[DuckDB]], and some custom C++ tool all agree to use Arrow's memory layout, they can pass data between eachother as a pointer (zero copy, zero serialization). This is the "==zero-copy IPC (inter-process communication==" that makes modern data pipelines fast!
	-  [[Pandas]], [[DuckDB]], [[Apache Spark|Spark]], [[Polars]], [[Google BigQuery|BigQuery]] etc. cna all speak Arrow natively.

==Arrow is NOT a file format, it is a MEMORY FORMAT.==
- [[Apache Arrow|Arrow]] format: The specification (language agnostic)
- [[Apache Arrow|PyArrow]]: Python bindings
- [[Apache Arrow|Arrow Flight]]: The RPC protocol for sending Arrow data over a network efficiently
- [[Apache Arrow|Arrow IPC]]: The file/stream format for persisting Arrow datas

The "[[Modern Data Stack]]" ([[DuckDB]] + [[Polars]] + [[Apache Parquet|Parquet]] + [[Apache Arrow|Arrow]]) is fast largely because data moves between tools without ever being deserialized, it's just passing points to shared memory buffers.
- Think of it as ==USB-C for data==, instead of every library inventing its own incompatible cable.

```python
import pandas as pd
import pyarrow as pa
import duckdb

# Example Earth observation-style tabular data
df = pd.DataFrame({
    "tile_id": [101, 102, 103],
    "ndvi": [0.72, 0.64, 0.81],
    "elevation_m": [120, 340, 210]
})

# Creates a coluimnar- in-memory representation with contiguous arrays, efficient CPU cache access, SIMD-friendly layout, and langauge-independent memory format
arrow_table = pa.Table.from_pandas(df)

# Now we can query Arrow memory directly in DuckDB!
# No copying rows around, just shared memory.
result = duckdb.query("""  
    SELECT  
        tile_id,  
        ndvi  
    FROM arrow_table  
    WHERE ndvi > 0.7  
""").to_df()  

```


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


![[Pasted image 20260425114257.png]]


## GeoArrow
- [[GeoArrow]] is to Arrow what [[GeoParquet]] is to Parquet -- just a specification for encoding geometry in Arrow's columnar format.
- Geometry operations can be vectorized over Arrow arrays using [[SIMD]], which is dramatically faster than deserializing [[Well-Known Binary|WKB]]s row-by-row.



