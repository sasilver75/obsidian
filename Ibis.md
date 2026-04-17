A Python dataframe library.
- Provides a [[Pandas]]-like Python API that compiles down to SQL and runs against a backend database or query engine (e.g. [[DuckDB]], [[BigQuery]], [[Snowflake]], [[PostgreSQL|Postgres]], etc.)
- You ==write Python expressions against a dataframe-like API==, and ==Ibis generates the SQL==, and the database executes it.

[[Ibis]] + [[DuckDB]] is a popular combination for analyzing large geospatial datasets locally without spinning up a full Postgres/PostGIS stack.
- It can work with [[GeoParquet]] and spatial operations at scale.
- For data science workflows, it lets you stay in Python while pushing computation to the database.

