---
aliases:
  - Replica
  - Read Replica
---
(Some of these comments are from "Jordan has no life," which might be bullshit)

Replication Benefits:
- Increasing the read (and possibly write) throughput.
- Increasing availability of the database system.

Replication can either by **Synchronous** or **Asynchronous**
- **Synchronous Replication** provides [[Strong Consistency]]; All replicas process the write before the leader node confirms the write to the client.
	- Not always viable, because if a Replica is unavialalbe, you can't serve writes.
- **Asynchronous Replication** is when the leader confirms the write after it writes to itself, and sometime later replicates data to its folllowers. This results in [[Eventual Consistency]], but higher throughput.


**Single Leader Replication**
- All writes go to a "leader" database, which are (e.g.) asynchronously replicated to "follower" databases. 
- Reads can come from either the leader or the follower.
- Pros: Simple to implement
- Cons: Write throughput limited to the leader replica. New leader election might incur temporary downtime. It's possible that the follower that becomes the next leader doesn't have all the data that the leader had on it, and we may lose some writes.

**Multi-Leader Replication**
- For a given key, you can have multiple leaders that can serve reads and writes for it.
- Pros: Increased write throughput, especially across datacenters.
- Cons: Write conflicts when the leaders synchronize with eachother, if the same key received ~simultaneous writes.
- Ways to resolve conflict:
	- [[Last Write Wins]]: Give everyone a timestamp
		- Note: Timestamps in distributed systems aren't very trustable because of [[Clock Skew]]/Drift. There's no guarntee the write you do pick is **actually** the one that occurred later. You basically lose the write that loses this comparison, by the way.
	- "Storing Siblings":
		- Leaders can identify when they have conflicting concurrent writes and store both of them. Later, a user can manually merge them together to go back to one copy. We can perform concurrent write detection via version vectors. Later, when someone reads the data, we see both the writes, somehow merges them back together, and writes to the database.
			- A version vector is a small array of how many writes have been processed by each leader (e.g. [A=1,B=2]).
				- When we merge version vectors, we keep the highest values for each entry in the vector. 
				- If one vector is strictly greater than the other at all positions, it's not considered a concurrent write.
		- Some databases ssupport merging automatically conflicting data for you!
			- For Counter, Set, Sequence objects. These may not support more complicated logic, however.

**Leaderless Replication**
- Writes and reads go to many nodes are once!
- Pros: No leader, so data should always be available to write and nodes should always be available to write to.
- Cons: Longer tail latencies, bottlenecked by the last replica to respond on write and reads.
	- If I need to wait for 3 nodes to respond before I can say a write is completed, that bottlenecks us.
	- So how many do we need to write to and read from before we consider a read/write as valid?
- **Replication [[Quorum]]s**
	- If we have N replicas, and write to W, and read from W, we want to be sure: W+ R > N, if so we have a quorum!
	- Quorums ensure that all reads and writes overlap on at least one replica.
- Handle conflicts again; can use version vectors or timestamps to determine which version of data is correct on reads.




Often: Replicas send data to one another via a "replication log," which is very similar to a [[Write-Ahead Log]].
- It basically just has all of the incremental operations being performed on the database.