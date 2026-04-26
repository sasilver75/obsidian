---
aliases:
  - Connection Pooling
---
Every time you query a database, you need a connection.

Opening a raw [[Transport Control Protocol|TCP]] connection to to [[PostgreSQL|Postgres]] takes ~5-10ms (expensive) if you do it on every request

Connection pooling keeps a pool of open connections that are used across requests.

[[SQLAlchemy]] manages this pool automatically:
```python
engine = create_async_engine(
    settings.database_url,
    pool_size=5,      # keep 5 connections open at all times
    max_overflow=10,  # allow up to 10 extra connections under load (total: 15)
)
```

When a request calls `get_db()`, it checks out a connection from the pool.
When the request ends, the connection is returned to the pool (not closed), where it is reused by the next request.

PostgreSQL has a connection limit of ~100. Each service that connects needs its own pool. So the ==application instances/consumers that are talking to PostgreSQL area the ones that manage their own pools, and they must consume connections conscientiously.==




