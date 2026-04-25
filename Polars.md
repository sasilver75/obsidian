Polars is a DataFrame library like [[Pandas]] written in Rust, designed from the ground up for performance.
- Columnar storage
- Vectorized execution, processing data in batches using [[SIMD]]
- Lazy evaluation: You build a query plan and Polars optimizes it before execution ([[Predicate Pushdown]], [[Projection Pushdown]], etc.)
- Parallelism: Multi-threaded by default, splits working across all cores.
	- [[Pandas]] is single-threaded, in comparison.


```python
 # Eager — executes immediately, like pandas
 # read_parquet() -> DataFrame (eager); immediatelyl reads the file into memory.
 # Every operation after this executes immediately and returns another DataFrame.
  df = pl.read_parquet("data.parquet")
  df.filter(pl.col("age") > 30).select(["name", "age"])

  # Lazy — builds a query plan, executes on .collect()
  # scan_parquet() -> LazyFrame; Every operation just adds to the query plan, nothing execs till collect()
  # Only reads your data into memory when you explicitly trigger execution, until then, it's just a query plan.
  df = pl.scan_parquet("data.parquet")
  df.filter(pl.col("age") > 30).select(["name",
  "age"]).collect()
```


# Comparison
- With Pandas:
	- Pandas is written in Python/C, Polars in Rust
	- Pandas is single-threaded, Polars multi-threaded
	- Pandas memory layout is mixed, and Polars is columnar, using [[Apache Arrow]]
	- Pandas does NOT support lazy evaluation, Polars does.
- With DuckDB
	- Similar performance characteristics, the different is mostly ergonomics: Polars is a DataFrame API, feeling more like pandas, while DuckDB is SQL. 
	- They can interoperate directly since Polars uses [[Apache Arrow|Arrow]] internally and DuckDB speaks Arrow natively.

![[Pasted image 20260425125535.png]]