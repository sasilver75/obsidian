---
aliases:
  - RLS
---

A [[PostgreSQL|Postgres]] feature where access rules are enforced on individual rows o na table, not just on the table as a whole.

Instead of "user can SELECT from `events`" (table-level), you write policies like:
- "user can SELECT an `events` row only if its location is unfuzzed OR they have a row i n`commits` for that event"
- "user can UPDATE an `events` row only if they are the host"

Postgres evaluates these policies on every query and can silently filter out the rows that the user shouldn't see, without your application code having to remember to add WHERE clauses.
- [[Supabase]] loves this, because their model lets the iOS app talk directly to Postgres via [[PostgREST]]. RLS makes this safe: without it, any client with a DB connection could read everything. With it, the database itself enforces "Attendees see real location, non-Attendees see fuzzed"