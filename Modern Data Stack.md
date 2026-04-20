A loosely-defined collection of Cloud-Native tools that became the dominant paradigm for data engineering ~2018-2023.

## "Classic" Modern Data Stack
The "Classic" modern data stack was:
```
Sources → Ingestion → Data Warehouse → Transformation → BI/ML                    (APIs)    (Fivetran)   (Snowflake)       (dbt)        (Looker)
```
- Use [[Fivetran]] or [[Airbyte]] to pull data from 100s of sources, land it raw.
- [[Snowflake]]/[[Google BigQuery|BigQuery]]/[[Amazon Redshift|Redshift]]: Cloud data warehouse, infinitely scalable
- [[dbt]]: Write SQL transforms, version-controlled, with testing/docs
- Looker/Tableau: BI layer on top

Big idea: Separate storage from compute, pay-per-query, no infrastructure to manage.


# "Modern" Modern Data Stack
- The "classic" MDS got expensive fast and created complexity, so a newer wave pushed back:
	- [[DuckDB]] instead of Snowflake for medium-scale workloads
	- [[Polars]] instead of Pandas
	- [[Apache Parquet|Parquet]] on S3 instead of a managed warehouse
	- [[dbt]] still survives (it won)
- The realization is that you don't need Snowflake until you *really* need Snowflake. A lot of "big data" problems are actually really medium-data problems that DuckDB can handle on a laptop.
- The truth is that the classic MDS was somewhat-hype driven (as perhaps is this "Modern" one):
	- Snowflake bills can get shockingly large
	- You end up with 8 SaaS tools with 8 sets of credentials, monitoring, and failure modes.
	- "Modern" became a marketing word more than a technical one.
- At this point, the stack is fragmenting and there's no single consensus anymore, but the `dbt + columnar storage + SQL-first` philosophy stuck. The debate is mostly about WHERE the compute runs (managed warehouse vs local/embedded engines like DuckDB).




_____________

The "[[Modern Data Stack]]" ([[DuckDB]] + [[Polars]] + [[Apache Parquet|Parquet]] + [[Apache Arrow|Arrow]]) is fast largely because data moves between tools without ever being deserialized, it's just passing points to shared memory buffers.

```python
import duckdb                                                                   import polars as pl                                                               

# DuckDB reads Parquet directly from disk (no Python involved)                                    
arrow_result = duckdb.sql("""
SELECT category, avg(price) as avg_price, count(*) as n                           FROM 'sales.parquet'                                                              WHERE year = 2024                                                                 GROUP BY category
""").arrow()  # returns an Arrow Table                                       

# Hand to Polars — zero copy, just a pointer to the same buffer                   
df = pl.from_arrow(arrow_result)                                             

# Continue in Polars                   
df.filter(pl.col("n") > 100).sort("avg_price", descending=True)
```
- Above, under the hood:
	- DuckDB's C++ engine scans the Parquet file (sales.parquet; also columnar, so it can skip column/row groups appropriately)
	- SQL query executes entirely in C++, Python is just waiting
	- `.arrow()` returns an Arrow Table, memory owned by DuckDB's C++ buffers
	- `pl.from_arrow()` gets a pointer to those same buffers (no data copied)
	- Polars operates on the same memory in its own Rust engine
- Python here is essentially the orchestration layer; actual data never really passes through it.

