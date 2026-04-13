DuckDB is an in-process analytical database, running inside your Python process, rather than as a separate server.

DuckSpatial adds geometric types on top.

It's excellent for offline analysis, Jupyter notebooks, running SQL against [[Apache Parquet|Parquet]] files on disk, single-machine batch jobs, etc. It's fast at column-oriented aggregation.

It's not a server though -- it has no concurrent multi-client access, no persistent network connection pool, and no long-running process that multiple clients can talk to simultaneously -- if you have two processes needing the same data, DuckDB's architecture fights you,.