Python needs a ==driver== to speak to the [[PostgreSQL]] wire protocool.

Two common drivers are used, for different purposes:
- [[asyncpg]]
	- A fast, async-native PostgreSQL driver, commonly used by FastAPI.
	- `
```
postgresql+asyncpg://user:pass@host/dbname
````
- [[psycopg]]
	- 