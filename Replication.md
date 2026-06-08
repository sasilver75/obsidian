---
aliases:
  - Replica
  - Read Replica
---
(Some of these comments are from "Jordan has no life," which might be bullshit)

Replication Benefits:
- Increasing the read (and possibly write) throughput.
- Increasing availability of the database system.
- Useful for [[Disaster Recovery]]
- Used in [[Change Data Capture]] (CDC), e.g. to populate a secondary [[Full-Text Search Index]]

# Timing

Replication can either by **Synchronous** or **Asynchronous**
- **[[Synchronous Replication]]** provides [[Strong Consistency]]; All replicas process the write before the leader node confirms the write to the client.
	- Not always viable, because if a Replica is unavailable, you can't serve writes.
- **[[Asynchronous Replication]]** is when the leader confirms the write after it writes to itself, and sometime later replicates data to its followers. This results in [[Eventual Consistency]], but higher throughput.
	- Replication lag can vary; asynchronous replication often means that replicas can serve stale reads, which breaks [[Read-your-Writes Consistency]] unless you route fresh reads to the leader.
	- There's a Failover risk too regarding lost writes; async replication can result in recently-acknowledged writes being lost if the promoted replica was behind.   


# Topology

**[[Single-Leader]] Replication**
- All writes go to a "leader" database, which are (e.g.) asynchronously replicated to "follower" databases. 
- Reads can come from either the leader or the follower.
- Pros: Simple to implement
- Cons: Write throughput limited to the leader replica. New leader election might incur temporary downtime. It's possible that the follower that becomes the next leader doesn't have all the data that the leader had on it, and we may lose some writes.
[[Split Brain]] can happen if there are two leaders elected after a network partition; needs [[Distributed Consensus|Consensus]]/fencing/leader election.

**[[Multi-Leader]] Replication**
- For a given key, you can have multiple leaders that can serve reads and writes for it.
- Pros: Increased write throughput, especially across datacenters.
- Cons: Write conflicts when the leaders synchronize with eachother, if the same key received ~simultaneous writes.
- Ways to resolve conflict:
	- [[Last Write Wins]]: Give everyone a timestamp
		- Note: Timestamps in distributed systems aren't very trustable because of [[Clock Skew]]/Drift. There's no guarantee the write you do pick is **actually** the one that occurred later. You basically lose the write that loses this comparison, by the way.
	- "Storing Siblings":
		- Leaders can identify when they have conflicting concurrent writes and store both of them. Later, a user can manually merge them together to go back to one copy. We can perform concurrent write detection via version vectors. Later, when someone reads the data, we see both the writes, somehow merges them back together, and writes to the database.
			- A version vector is a small array of how many writes have been processed by each leader (e.g. \[A=1,B=2]).
				- When we merge version vectors, we keep the highest values for each entry in the vector. 
				- If one vector is strictly greater than the other at all positions, it's not considered a concurrent write.
		- Some databases support merging automatically conflicting data for you!
			- For Counter, Set, Sequence objects. These may not support more complicated logic, however.

**[[Leaderless]] Replication**
- Writes and reads go to many nodes are once!
- Pros: No leader, so data should always be available to write and nodes should always be available to write to.
- Cons: Longer tail latencies, bottlenecked by the last replica to respond on write and reads.
	- If I need to wait for 3 nodes to respond before I can say a write is completed, that bottlenecks us.
	- So how many do we need to write to and read from before we consider a read/write as valid?
- **Replication [[Quorum]]s**
	- If we have N replicas, and write to W, and read from R, we want to be sure: W+ R > N, if so we have a quorum!
	- Quorums ensure that all reads and writes overlap on at least one replica.
- Handle conflicts again; can use version vectors or timestamps to determine which version of data is correct on reads.


# Mechanism

[[Physical Replication]]
- Copies the database's low-level storage/log representation, usually [[Write-Ahead Log]] (WAL) records.
- Best for hot standby, read replicas, failover, and disaster recovery. 
- Typically requires the same DB engine/version/storage format, whole cluster/database, replicas often read-only.


[[Logical Replication]]
- Decodes the log into semantic row changes: inserts, updates, deletes.
- Best for [[Change Data Capture|CDC]], selective table replication, cross-version migrations, search indexes, analytics sinks, and event-driven projections.
- More flexible, but schema changes, backfills, sequences, DDL, and conflicts need care.


____________

# How does routing to replicas work?

Q: From the perspective of an application server instance that needs to get some data from the database, how does it actually route to one of the replicas?

A: This might be implemented in one of several ways!

1. Application-level Routing
	- The app has two database addresses configured (e.g. primary-postgres:5432, replica-postgres:5432), so when the app needs to database, it chooses which to connect to.
2. A database proxy or [[Load Balancing|Load Balancer]]
	- The app thinks it is connected to one host (`postgres://app:pass@read-db.internal:5432/mydb), but the proxy/load balancer chooses a backend replica. 
3. Managed database reader endpoints
	- Managed databases often provide special endpoints, e.g. `mydb-writer.cluster.example.com` for the primary/leader, and `mydb-reader.cluster.example.com` for the replicas. 
	- Your application would store these as config variables and use them appropriately, with the cloud provider handling replica health checks and connection-level load balancing.
4. Driver or ORM support
	- Some database drivers and ORMs support multiple hosts, read/write splitting, or "read-only transaction" routing.


Q: I'm not really satisfied with that. There's two parts, there's turning a logical address into a routable address, and hten actually routing a message to it.

A: Yes, there are two different routing steps:

1. Choose the database endpoint
2. Route packets to that endpoint

If we have `READ_DB = postgres-read.internal:5432`

When the app wants a replica read:
1. App chooses `postgres-read.internal`
2. App asks DNS/service discovery: "What IP is this?"
3. DNS returns something like 10.0.4.25
4. App opens a TCP connection to 10.0.0.4.25:5432
5. The OS sends packets towards 10.0.4.25 using its routing table
6. Routers/VPC networking/Kubernetes networking delivers the packets
7. A Postgres server or proxy receives the TCP connection
8. SQL bytes flow over that connection

So application-level routing really means: The app chooses the logical address, and then the network stack routes to the resolved IP address.

==Note that `postgres-read.internal` might resolve to many things.== 

Case 1: [[Domain Name Service|DNS]] might point directly at replicas:
```
postgres-read.internal -> 10.0.1.11
postgres-read.internal -> 10.0.1.12
postgres-read.internal -> 10.0.1.13
```
So the app gets one or more IPs to choose from, and it chooses a TCP connection to one of them.

Case 2: [[Domain Name Service|DNS]] points at a load balancer
`postgres-read.internal -> 10.0.4.25`
- But 10.0.4.25 isn't a database, it's a [[Load Balancing|Load Balancer]]! Our app chats to this, and our Load Balancer chooses the actual replica

Case 3: DNS points at a managed reader endpoint
For example: `mydb-reader.cluster.interal`
The cloud provider then resolves or forwards this to a hostname