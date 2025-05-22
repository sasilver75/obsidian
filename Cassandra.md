SDIAH: https://www.hellointerview.com/learn/system-design/deep-dives/cassandra

Cassandra is one of the most versatile/popular databases to have in your toolbox. 
- It was originally built by Facebook to support its rapidly scaling inbox search feature.
- Cassandra has been adopted by countless companies to rapidly scale data storage, throughput, and readback.

==Cassandra is an open-source, distributed [[NoSQL Database]] that implements a **partitioned [[Wide-Column]] storage model with [[Eventual Consistency|Eventually Consistent]] semantics**==
- It runs in a cluster and can [[Horizontal Scaling|Horizontally Scale]] via commodity hardware.
- It combines elements of [[DynamoDB]] and [[Bigtable]] to handle massive data footprints, query volume, and flexible storage requirements. 


# Cassandra Basics

### Data Model
![[Pasted image 20250521134107.png]]
- Terminology:
	- ==**Keyspace**==: **Keyspaces** are basically data containers, and can be likened to "databases" in relational systems like Postgres or MySQL. They contain many **Tables**, and are responsible for owning configuration information about the tables. For example, Keyspaces have a configured replication strategy for managing data redundancy/availability. The keyspace also owns any **user-defined-types (UDTs)** that you might make to support your use case.
	- ==**Table**==: A table is a container for your data, in the form of **rows**. It has a name and contains configuration information about the data that's stored within it.
	- **==Row==**: A row is a container of data. It is represented by a primary key, and contains **columns**.
	- **==Column==**: A column contains data belonging to a **row**. 
		- ==**A column is represented by a name, a type, and value corresponding to the value of that column for a row.**==
		- ==Not all columns need to be specified per row in a Cassandra table==
		- Cassandra is a [[Wide-Column]] database, so the **specified columns can vary per row in the table**, making Cassandra **more flexible** than something like a relational database.
		- Every column has **timestamp metadata** associated with it, denoting when it was written. When a column has a write conflict between replicas, it is resolved via [[Last Write Wins]] strategy.
- ==At the most basic level, **you can liken Cassandra's data structures to a large JSON!**==
```
{
  "keyspace1": {
    "table1": {
      "row1": {
        "col1": 1,
        "col2": "2"
      },
      "row2": {
        "col1": 10,
        "col3": 3.0
      },
      "row3": {
        "col4": {
          "company": "Hello Interview",
          "city": "Seattle",
          "state": "WA"
        }
      }
    }
  }
}
```
- Cassandras supports a plethora of types, including user-defined types and JSON values, which makes Cassandra very flexible as a data store for both flat and nested data.

### Primary Key
- One of the ==most important constructs in Cassandra is the **"primary key"**== of the table!
- ==Every row is represented uniquely by a primary key==, and a primary key ==consists of one or more **partition keys**==, and ==may include **clustering keys!**==
	- **==Partition Key==**: **One or more** columns that are used to determine which **[[Partition|Partitioning]]** the row is in.
	- **==Clustering Key==**: **Zero or more** columns that are used to determine the **sorted order** of rows in a table. 
		- Data ordering is important, depending on one's data modeling needs, so Cassandra gives users control over this via the clustering keys.

When you create a table in Cassandra via the [[Cassandra Query Language]] (CQL) dialect, you specify the primary key as part of defining the schema:
```sql
-- Primary key with partition key a, no clustering keys
CREATE TABLE t (a text, b text, c text, PRIMARY KEY (a));

-- Primary key with partition key a, clustering key b ascending
CREATE TABLE t (a text, b text, c text PRIMARY KEY ((a), b))
WITH CLUSTERING ORDER BY (b ASC);

-- Primary key with composite partition key a + b, clustering key c
CREATE TABLE t (a text, b text, c text, d text, PRIMARY KEY ((a, b), c));

-- Primary key with partition key a, clustering keys b + c
CREATE TABLE t (a text, b text, c text, d text, PRIMARY KEY ((a), b, c));

-- Primary key with partition key a, clustering keys b + c (alternative syntax)
CREATE TABLE t (a text, b text, c text, d text, PRIMARY KEY (a, b, c));
```
- Above: Again, the partition key determines which partition our data ends up on, and the clustering key determines the sorted order of rows in a table!


## Key Concepts
- You're going to want to know more than just how to use Cassandra; you'll want to be able to explain how it works in case your interviewer asks pointed questions, or you might want to be able to deep-drive into data storage specifics, scalability, query effficiency, etc. -- all of which deeply affect your design. 
#### Partitioning
- One of the most fundamental parts of Cassandra is how it partitions data.
- Cassandra partitioning techniques are extremely robust and worth understanding **generally** for system design, in case you want to employ them in other areas of your designs (caching, load balancing, etc.)
- In order to partition data successfully, Cassandra makes use of [[Consistent Hashing]], a fundamental technique in distributed systems to partition data/load across machines in a way that prioritizes the evenness of distribution while minimizing the re-mapping of data if a node enters or leaves the system.

In a **traditional hashing scheme**:
- A number of nodes are chosen
- A Node is determined to store a record by taking hash(recordPartitionKey)%numNodes to determine the node to allocate the record to.
- **Problems with this strategy:**
	- If the number of buckets changes (node added or removed), then *a lot of values* will be assigned to new nodes! In a distributed system like a database, this means that data would have move between nodes in excess. (e.g. if you had Nodes A,B,C and added a node D, this remapping would involve A moving some data to B, which doesn't really make sense, given that D is the one that joined the cluster
	- If you're unlucky, there might be a lot of values that get hashed to the same node, resulting in uneven load between nodes.

In **Consistent Hashing:**
- Rather than hashing a value and running a modulo to select a node, we hash a a value to a range of integers that are visualized on a ring:
![[Pasted image 20250521141342.png|500]]
- The ring has nodes on it mapping to specific values. When a value is hashed, we hash it to an integer on the ring, and then the ring is walked clockwise to find the first value corresponding to a node. The value is then stored on that node.
- This ==prevents excess remapping of values if a node enters or leaves the system==, because a node entering or leaving only affects *one adjacent node*! 
- **PROBLEM:** This doesn't address the issue of uneven load between nodes! To address this, we use multiple **virtual nodes** on the ring that all map to the same phyiscal node in the system. These **VNodes** help distribute load over the cluster more evenly, and also **allows for the system to take advantage of heterogenously-equipped nodes in the cluster; more powerful machines can have more VNodes.**
	- ![[Pasted image 20250521141604.png]]


### Replication
- In Cassandra, partitions of data are [[Replicate]]d to nodes on the ring, enabling it to skew to being ==extremely available== for system designs that need that feature.
- **KeySpaces** (think: "Databases" in Cassandra) have replication configurations specified that affect the way Cassandra replicates data.
- At a high level, ==Cassandra chooses what nodes to **replicate** data to by scanning clockwise from the VNode that corresponds to a hashed value in a consistent hashing scheme==.
	- ![[Pasted image 20250521141754.png]]
	- So if a Cassandra KeySpace is set to replicate data to 3 nodes in total, it will hash a value to a node and then scan clockwise to find 2 additional VNodes to serve as replicas!
		- Importantly, ==Cassandra skips any VNodes that are on the same physical node, so that multiple replicas don't go down when a single physical node goes down==
- ==Cassandras has **TWO replication strategies that it can employ!**==
	1. **==NewtworkToplogyStrategy==**: The strategy ==recommended for production== and is ==data center/rack aware== so that data replicas are stored across potentially many data centers in case of an outage.
		1. It allows for replicas to be stored on distinct racks in case a rack in a data center goes down.
		2. The main goal with this configuration is to establish enough physically-separate replicas to avoid many replicas being affected by some real-world outage or incident.
	2. **==SimpleStrategy==**: A simple strategy, merely determining replicas via scanning clockwise. Useful for simple deployments and testing.

```cql
-- 3 replicas
ALTER KEYSPACE hello_interview WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 3 };

-- 3 replicas in data center 1, 2 replicas in data center 2
ALTER KEYSPACE hello_interview WITH REPLICATION = {'class' : 'NetworkTopologyStrategy', 'dc1' : 3, 'dc2' : 2};
```
Above: Specifying replication strategies for Cassandra **KeySpaces** using [[Cassandra Query Language|CQL]]


### Consistency
- Like any distributed system, Cassandra is subject to the [[CAP Theorem]]
	- Cassandra gives uses flexibility over consistency settings for reads/writes, which allows Cassandra users to "tune" their consistency/availability tradeoff.
- ==Cassandra does not offer transaction support or any notion of [[ACID]] guarantees. It only supports atomic and isolated writes at the row level in a partition, but that's about it.==
	- It only supports atomic and isolated writes at the **row level** in a [[Partition]], but that's about it.  More [here](https://docs.datastax.com/en/cassandra-oss/2.2/cassandra/dml/dmlTransactionsDiffer.html).
- ==**Cassandra allows you to choose from a number of "consistency levels" for both reads and writes, which are required node response numbers for a write or read to succeed.==**
	- These enforce different consistency and availability behavior depending on the combination used.
	- These range from ONE (where a single replica needs to respond) to ALL (where all replicas must respond): More [here](https://cassandra.apache.org/doc/latest/cassandra/architecture/dynamo.html#tunable-consistency)
	- One consistency level to understand is [[Quorum]], which require a majority (n/2+1) of replicas to respond.
		- ==Applying Quorum to both reads and writes guarantees that writes are visible to reads, because at least one overlappping node is guaranteed to participate in both a write and a read.==
![[Pasted image 20250521144107.png|500]]
- Typically, Cassandra aims for [[Eventual Consistency]] for all consistency levels, where all replicas have the latest data, assuming enough time passes.

### Query Routing
- ==***Any Cassandra node can service a query from the client application*** because all nodes in Cassandra can assume the role of a query "**coordinator**"==.
	- Nodes in Cassandra each know about other alive nodes in the cluster.
	- They also share cluster information via a [[Gossip]] protocol!
	- Nodes can determine where data lives in the cluster via performing a [[Consistent Hashing]] calculation and by knowing the replication strategy/consistency level configured for the data.
	- When a client issues a query, it selects a node who becomes the coordinator, and the coordinator issues queries to nodes that store the data (a series of replicas)
		- ((I'm sure the book-keeping for this is very complicated, since you might know the replicas that should have gotten the write when it occurred, but since then perhaps nodes have entered or left the cluster, perhaps removing or inserting VNodes into the hash ring between the designated primary node and the replica nodes)).







