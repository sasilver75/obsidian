---
aliases:
  - LSI
---
See also: [[Global Secondary Index]]
In the case of a [[Sharding|Shard]]ed database...

A secondary index stored on the same shard/partition as the data they index.
- Typically share the same partition key as the main table, but may have a different sort key.
- Since both data and index are on the same shard, queries can maintain strong consistency.
- Can only query data within the same partition.

Advantages:
- Strong consistency: No eventual consistency issues, since data and index are co-located. 
- Lower latency: No cross-shard communications needed for index lookups.
- Simpler transactions: ACID properties easier to maintain within a single shard.

Disadvantages:
- Limited Query Flexibility; Can only query within teh same partition key
- Uneven Distribution: If queries aren't evenly distributed among partitions, some shards may become hot spots.
- Storage Overhead: Each shard must maintain its own copy of the index.

How you use it (in a situation where CustomerId is our partition key, and each partition has a LSLI by order_date):
- Query: "Show me all orders for Customer 123 sorted by date"
- System: Goes directly to Server A, uses the LSI to quickly find orders by date
- Result: Fast, consistent results because everything is on one server.
What you CAN'T do:
- Query: "Show me all orders from last week across all customers"
- Problem: Would need to check every server, since data is split by customer.


(Choose LSLI when users mostly ask about their own data, you need immediate consistency, or when you have simple/predictable access patterns.)
(Choose GIS when you need analytics/reporting across all users, search functionality across the entire dataset, multiple different ways to access the same data, if you can handle slight delays in data updates.)
(LSLI keeps things simple and fast for individual users, and GSI enables powerful queries across your entire dataset, but with added complexity)