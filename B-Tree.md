
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


