---
aliases:
  - Cursor-Based Pagination
  - Offset-Based Pagination
---

# Offset-Based Pagination
- "Skip 150 rows, return the next 50."
- Good when users need random access, or against stable-ish data.
- Uses page numbers and page sizes: `GET /orders?page=4&page_size=50`
- Usually implemented with a SQL `SELECT ... LIMIT 50 OFFSET 150;`
```sql
  SELECT *
  FROM orders
  ORDER BY created_at DESC
  LIMIT 50 OFFSET 150;
```

# Cursor-Based Pagination
- "Return the next 50 rows after this last seen position."
- Good when data is large, changing, or feed-like (activity feeds, chat messages, event logs)
- Uses a marker from the last item seen: `GET /orders?limit=50&cursor=eyJjcmVhdGVkX2..."`
	- This `eyJjcmVhdGVkX2` is a [[Base64]]-encoded `{"created_at": 2026-06-01T100:00:00Z, "id": "ord_123"}`, which is used below to write the SQL query that basically says "Continue from where I left off"
- Often implemented as keyset pagination:
```sql
SELECT *
FROM orders
WHERE (created_at, id) < (:last_created_at, :last_id)
ORDER BY created_at DESC, id DESC
LIMIT 50;
```
- A good cursor usually includes: sort key + tie-breaker.

You start cursor pagination by making the first request *without a cursor*
- `GET /orders?limit=50`

Q: Why do you need both `last_created_at` and `last_id`? Why not just one?
A: Because `created_at` alone might not be unique! Imagine that you had
```
id=101 created_at=10:00:00
id=102 created_at=10:00:00
id=103 created_at=10:00:00
id=104 created_at=09:59:00
```
If page ends at `id=102 created_at=10:00:00` and your next query is only:
```
WHERE created_at < '10:00:00'
ORDER BY created_at DESC
LIMIT 50
```
Then you would skip `id=103` and `id=104`!

Yan can use one field only if that field is: unique, stable, indexed, sorted in the exact order you want (e.g. a monotonically increasing `id`)