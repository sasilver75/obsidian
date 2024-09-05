[[HyDE]] starts with a real *query* and generates a hypothetical document that would be relevant to query. Then it finds *real* documents that are close to our hypothetical document in embedding space.

In [[Reverse HyDE]], we instead start with a real *document* and generates a hypothetical *query* that would be highly likely to retrieve that document.
- We can take the encoding/embedding of this hypothetical query to find similar *actual* queries, or to understand what kinds of queries might lead to this document. It's particularly useful in scenarios where understanding how documents might be discovered or queried is important.

