Unlike [[View]]s, which are just stored queries than run live on every SELECT, materialized views are ==physically stored snapshots!== They're "copies" of some source data, typically with some transformation applied.

Whether a materialized view "automatically updates itself" is up to the database system.


```sql
REFRESH MATERIALIZED VIEW [ CONCURRENTLY ] _`name`_
    [ WITH [ NO ] DATA ]
```
- `REFRESH MATERIALIZED VIEW` completely replaces the contents of a materialized view.
	- It's a ==full recomputation==; Postgres might scan millions of rows from source tables, re-aggregate, and write the result. I should say: ==In [[PostgreSQL|Postgres]], there's no incremental/partial refersh, unlike in [[Snowflake]] or [[Google BigQuery]]==
- `CONCURRENTLY`: Refresh the materialized view without locking out concurrent selects on the materialized view. Without this option a refresh which affects a lot of rows will tend to use fewer resources and complete more quickly, but could block other connections which are trying to read from the materialized view.
	- Refreshing a materialized view may take seconds or minutes, depending on how much data you've got. Adding ==CONCURRENTLY keeps it non-blocking==, with Postgres (e.g.) rebuilding the materialized view in the background and atomically swapping it in. Reads against the old MV continue uninterrupted during the refresh.
