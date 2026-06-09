

When your query makes ==1== query to fetch a list of data, and then makes ==N== additional queries to fetch related data for each item in that list.


Example:

1 Query: `Fetch 100 posts`
Then N Queries:
- `Fetch comments for Post 1`
- `Fetch comments for Post 2`
- `Fetch commments for Post 3`
- ...

So instead of doing 1 or 2 efficient queries, you do 1 + 100 = 101 queries, which is why it's called ==N + 1==
- Each database query has overhead; one query might be cheap, but 100s or 1000s of tiny queries (and joins) can crush latency and increase database load.


# How do we "solve" the N+1 Problem?


The typical recommended fix is ==Eager Loading / Preloading==, where we simply fetch the related data up front:
```sql
SELECT posts.*, users.*
FROM posts
JOIN users ON users.id = posts.author_id
LIMIT 50;
```
This is usually the recommended first fix: turning many small queries into one or a few predictable queries.
We can combine this with [[Cache|Caching]] so we don't have repeated lookups of an expensive join always hitting the database.


In some situations, we can also [[Denormalization|Denormalize]] our storage:
- If for a list view of posts, we always need `posts.author_name`
	- And we have a `posts` and `users` table with a post.author_id FK that requires a join...
- We can instead just have a `posts.author_name` instead of always joining `users`.

This is good for read-heavy systems that need good latency, but the tradeoff is that writes become more complex because duplicated data must be kept in sync.



We can also change the API shape (this is sometimes called the Includes or Expansion pattern)
`GET /posts?include=author,comments`
- This can work for APIs where clients may or may not need related data.
	- More control, but more API complexity.











