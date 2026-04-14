---
aliases:
  - Database API
---
A Python specification (PEP 249) that defines a standard interface for all Python database drivers that they must implement.  It is a contract, with the idea that if every database driver exposes the same methods (`connect()`, `cursor()`, `execute()`, `fetchone()`, `fetchall()`, etc.), then higher-level tools like [[SQLAlchemy]] can talk to *any* database without needing to know the specifics of each driver, whether it's [[psycopg]] or [[asyncpg]].

Code -> SQLAlchemy -> DBAPI Driver -> PostgreSQL


The key objects

Connection — represents a session with the database:
```python
conn = driver.connect(host=..., user=..., password=...)
conn.commit()
conn.rollback()
conn.close()
```

Cursor — represents a single query execution:
```python
cursor = conn.cursor()
cursor.execute("SELECT * FROM sr_311 WHERE srnumber = %s", ("1-12345",))
row = cursor.fetchone()
rows = cursor.fetchall()
```





