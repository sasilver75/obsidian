---
aliases:
  - Log-Structured Merge Tree
---


Article referred to in SDIAH, which might be interesting to read: https://hackernoon.com/how-cassandra-stores-data-an-exploration-of-log-structured-merge-trees


BAD REPLACE ME
Writes first go to an in-memory BST, which is then flushed to immutable sorted tables on disk when it gets to big. Reads for a key are first performed from the tree, and if not present, we check each SSTable file from newest to oldest. Bloom filters arlso used!
Pros: Fast writes to in-memory!
Cons: Fairly slow reads due to having to check multiple different places for the value for a given query.