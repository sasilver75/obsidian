---
aliases:
  - GSI
---
See also: [[Local Secondary Index]]
In the case of a [[Shard]]ed database...

Global secondary indexes are indexes that span across multiple shards and can have a completely different partitioning scheme than the main table.
- Index data is distributed across multiple shards, potentially different frmo the main table.
- Can query across all partitions using the index.
- Often [[Eventual Consistency|Eventually Consistent]] due to the distributed nature.

Advantages:
- Query flexibility; Can query across all data regardless of the main table's partition key
- Better load distribution; Can distribute query load more evenly across shards
- Multiple access patterns: Support different query patterns than the main table

Disadvantages:
- Eventual Consistency; Updates to main table may not immediately reflect in the GSI
- Higher Complexity: More complex to maintain and manage
- Cross-Shard Optimizations: May require multiple network calls and coordination
- Storage Cost: Additional storage overhead across multiple shards



How you use it (in a situation where CustomerId is our partition key, and we have a GSI split by product_id):
```
Main Table:
Server A: Customer 123's orders
Server B: Customer 456's orders  
Server C: Customer 789's orders

GSI (by product_id):
Server X: All orders for Product "iPhone"
Server Y: All orders for Product "MacBook"
Server Z: All orders for Product "iPad"
```
- Query: "Show me all customers who bought iPhones his month"
- System: Goes to Server X (iPhone GSI), finds all iPhone orders regardless of which customer bought them.
- Result: Can see orders from customers 123, 456, 789 all in one place.



(Choose LSLI when users mostly ask about their own data, you need immediate consistency, or when you have simple/predictable access patterns.)
(Choose GIS when you need analytics/reporting across all users, search functionality across the entire dataset, multiple different ways to access the same data, if you can handle slight delays in data updates.)
(LSLI keeps things simple and fast for individual users, and GSI enables powerful queries across your entire dataset, but with added complexity)