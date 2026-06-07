

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











