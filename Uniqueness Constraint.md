---
aliases:
  - Composite Uniqueness Constraint
---


A database rule that says that no two rows in a table may have the same value, or the same combination of values, for specified columns.
- The main purpose of a uniqueness constraint is *correctness;* it prevents invalid state from entering the DB. But is also critical for things like the [[Inbox Pattern]], where we want to deduplicate effort.

From a user's perspective in [[PostgreSQL]], a uniqueness constraint is a named rule backed by a unique [[B-Tree]] index:
```sql
CREATE TABLE users(
	id bigserial PRIMARY KEY,
	email text UNIQUE
);
```
Postgres treats this roughly as: "Crate a constrained named something like `users_email_key` and create a unique index that can efficiently detect duplicate `email` values."
Later, if you insert another record with the same email, you get:
```
duplicate key value violates unique constraint
```

You can also create uniqueness constrains on multiple columns, called [[Uniqueness Constraint|Composite Uniqueness Constraint]]s:
```sql
CREATE TABLE memberships (
  user_id BIGINT NOT NULL,
  organization_id BIGINT NOT NULL,
  role TEXT NOT NULL,

  UNIQUE (user_id, organization_id)
);
```
This allows for:
```
user_id | organization_id
--------+----------------
1       | 10
1       | 11
2       | 10
```
But would reject a new row with `(1, 10)` for instance.


Some databases even let you create unique indexes directly, especially for advanced cases like partial uniqueness:
```sql
CREATE UNIQUE INDEX unique_active_email
ON users (email)
WHERE deleted_at IS NULL;
```
This means "An email must be unique among active users, but deleted users do not count."







