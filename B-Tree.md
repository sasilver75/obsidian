
A database [[Index]] that stores indexed values in sorted order inside a balanced page-oriented tree, so that the database can quickly find rows without doing a whole table scan.

Strictly, it's a "self-balancing multi-way search tree, where each node contains many sorted keys and child points, and all leaf nodes stay at the same depth."

Frequently, "B-Tree Index" often refers to a [[B+ Tree]]-like structure, where internal nodes guide the search, and leaf nodes contain the actual index entries.

Think of a B-Tree Index as a sorted map stored on disk:
```
email value                  row location
------------------------------------------------
aisha@example.com             page 104, slot 7
bruno@example.com             page 891, slot 2
maya@example.com              page 552, slot 9
zara@example.com              page 113, slot 4
```
But this map is too large to sit in one page, so the database stores the map as a tree of [[Page]]s.. each internal page says, roughly: "For keys in *this* range, go to *that* child page!"

- Root Page: The top of the tree.
- Internal Page: Routing pages containing separator keys and child pointers.
- Leaf Pages: Bottom-level pages containing sorted index entries.
- Row locations: Pointers from index entries to table rows, unless the database uses a [[Clustered Index]] layout where the leaf entries contain the actual row data itself.

A B-Tree Index is the database’s general-purpose ordered lookup structure. It is excellent for equality lookups, range queries, sorted access, and composite indexes whose leftmost columns match the query. Its main tradeoff is that faster reads come at the cost of extra storage and slower writes.

On a read, like:
```sql
SELECT ... WHERE email = 'maya@example.com'
```
1. Read the root index page
2. Compare `maya@example.com` against separator keys
3. Follow the correct child pointer
4. Repeat until reaching a leaf page
5. Find the matching index entry
6. Follow the row location to fetch the actual table row

Because each page contains many keys, each page can point to many child pages. This high *branching factor* keeps the tree shallow -- a B-Tree with billions of rows may still be only 3-5 levels deep.`

__________

For a B-Tree, nodes are typically the same size as dat pages (~4KB, 8KB, or 16KB, depending on the database)


![[Pasted image 20250520180143.png]]

B-Trees have become the default choice for most database indexes because they excel at everything databases need:
1. They maintain sorted order, making range queries and ORDER BY operations efficient.
2. They're ==self-balancing, ensuring predictable performance== even as data grows.
3. They minimize disk I/O by matching their structure to how databases store data.
4. They handle both ==equality searches== (email="X") and ==range searches== (age > 25) equally well.
5. They ==remain balanced even with random inserts and deletes==, avoiding the performance cliffs you might see with simpler data structures.


In Postgres, when you run:
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE
);
```
This creates **two** B-Tree indexes: One for the primary key and one for the unique email constraint! These B-Trees maintain sorted order, which is crucial for both uniqueness checks and range queries!
- DynamoDB's sort key is also implemented as a B-tree variant, allowing for efficient range queries within a partition.
- Even MongoDB's document model uses [[B+ Tree]] (a variant where all data is stored ***in*** leaf nodes) for its indexes!


