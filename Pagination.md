---
aliases:
  - Cursor-Based Pagination
  - Offset-Based Pagination
---

Offset-Based Pagination
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

Cursor-Based Pagination
- "Return the next 50 rows after this last seen position."
- Good when data is large, changing, or feed-like (activity feeds, chat messages, event logs)
- Uses a marker from the last item seen: `GET /orders?limit=50&cursor=eyJjcmVhdGVkX2..."`
- Often implemented as keyset pagination:
```sql
SELECT *
FROM orders
WHERE (created_at, id) < (:last_created_at, :last_id)
ORDER BY created_at DESC, id DESC
LIMIT 50;
```
- A good cursor usually includes: sort key + tie-breaker.

