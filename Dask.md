"Numpy/Pandas, but distributed"

Core idea:
- Dask mirrors the APIs you already know; `dask.dataframe` looks like [[Pandas]], and `dask.array` looks like [[NumPy]].
- You change almost nothing about your code, and Dask handles distributing it.


How it works:
- Breaks data into ==partitions== (chunks of a DataFrame, tiles of an Array)
- Builds a task graph of operations (lazily, like [[Apache Spark|Spark]])
- Executes the graph using a scheduler, either locally on multiple cores, or distributed across a cluster.

Strengths:
- Near-zero learning curve if you know Pandas/NumPy
- Works on a single laptop (multiple cores), with no cluster required.
- Lightweight, pure Python
- Good for "medium data" (10GB - 1TB)

Weaknesses:
- Slower than [[Apache Spark|Spark]] at massive scale
- Less mature ecosystem for SQL/Streaming