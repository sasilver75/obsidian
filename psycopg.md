Python needs a ==driver== to speak to the [[PostgreSQL]] wire protocool.

Two common drivers are used, for different purposes:
- [[asyncpg]]
	- A fast, async-native PostgreSQL driver, commonly used by FastAPI.
	- `postgresql+asyncpg://user:pass@host/dbname`
	- ==Asnyc-only==, so every query is `await`-able.
	- Very fast, with no overhead from the DBAPI layer.
	- *==Does not support synchronous use==!*
- [[psycopg]]
	- The =="standard" PostgreSQL driver== for Python, used commonly with [[Alembic]] migrations and [[Celery]] workers.
	- `postgresql+psycopg://user:pass@host/dbname`
	- ==Synchronous by default, with optional async support==.
	- ==Required by Alembic's migration runner==, which is inherently synchronous.




