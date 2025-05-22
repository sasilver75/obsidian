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
	- When a **client** issues a query, it **selects a node who becomes the** ***==coordinator==***, and the **coordinator issues queries to nodes that store the data** (a series of replicas)
		- ((I'm sure the book-keeping for this is very complicated, since you might know the replicas that should have gotten the write when it occurred, but since then perhaps nodes have entered or left the cluster, perhaps removing or inserting VNodes into the hash ring between the designated primary node and the replica nodes)).
![[Pasted image 20250521182627.png]]


### Storage Model
- Cassandra's storage model is important to understand because it's core to one of its strengths for system design: ==Write throughput!==
- Cassandra leverages a data structure called a [[LSM Tree|Log-Structured Merge Tree]] (LSM Tree) index to achieve this speed!
	- The LSM-Tree is used **in place of** a [[B-Tree]], which is the index of choice for most databases.
- **Cassandra opts for an approach that favors write speed over read speed!**
	- Every create/update/delete is a **new entry** (with some exceptions)
	- Cassandra uses the ordering of these updates to determine the "state" of a row.
		- If a row is created, then updated later, Cassandra will understand the state of the row by looking at the creation and then the update, versus just looking at a single row.
		- The same can be said for deletes, which can be thought of as "removal updates", with Cassandra writing a [[Tombstone]] entry for row deletions.
	- The **LSM Tree** enables Cassandra to efficiently understand the state of the row, while writing data to the database as almost entirely **==append-only writes==** (which are fast to do!).

The 3 constructs that are core to the [[LSM Tree]] index are:
1. **==Commit Log==**: This is basically a [[Write-Ahead Log]] (WAL) to ensure durability of writes for Cassandra nodes.
2. **==Memtable:==** An **in-memory**, sorted data structure that storse writes data. It is sorted by the primary key of each row.
3. **==[[SSTable]]==**: A "Sorted String Table". An **immutable** file **on disk** containing data that was flushed from a previous **Memtable**.

With all of these working together, writes to [[Cassandra]] look like this:
1. A write is issued for a node ((We know from before that this involves a client talking to any node, which acts a coordinator to determine the node to write to; it contacts that node to confirm the write, etc.))
2. That write is written to the **commit log** so that it doesn't get lost if the node goes down while the write is being processed, or if the data is only in the Memtable when the node goes down.
3. The write is written to the **Memtable**.
4. Eventually, the Memtable is flushed to disk as an immutable **SSTable** after some threshold size is hit, or some period of time elapses.
5. When a Memtable is flushed, any commit log messages are removed that correspond to that Memtable, to save space. These are superfluous, now that the Memtable has been durable stored on disk as an immutable SSTable.

![[Pasted image 20250521190219.png]]
- To summarize, a **Memtable houses recent writes, consolidating writes for a key into a single row**, and is occasionally flushed to disk as an immutable **SSTable**.
- A **commit log** serves as a [[Write-Ahead Log|WAL]] to ensure that data isn't lost if it's only in the Memtable and the node goes down.
- ((I assume that the write can be acked to the writer (e.g. coordinating node) when it's been appended to the WAL commit log))

When **==reading data==** for a particular key:
1. Cassandra first attempts to read from the **Memtable**, which will have the latest data.
2. If the Memtable doesn't have the data for the key, Cassandra will leverage [[Bloom Filter]]s to determine *which* **SSTables** on disk might have the data.
	1. Each **SSTable** has its own **Bloom Filter** kept in memory; These are probabilistic data structures that tell us if a key is either (DEFINITELY NOT) or (MAYBE) in a table. Because they're all in-memory, we can check all of the bloom filters quickly.
3. It then reads the **SSTables** in order from newest to oldest to find the latest data for the row.
	1. I assume for each potentially-relevant SSTable, we read it into memory and then search through it.
4. Of note, the data in **SSTables** is sorted by primary key, making it easy to find a particular key.

Building on the above foundation, there's two additional concepts to internalize:
- ==**Compaction**==: To prevent bloat of SSTables with many row updates/deletions, Cassandra will run [[Compaction]] to consolidate data into a smaller set of SSTables, which reflect the consolidated state of data. Compaction also removes rows that were deleted, removing tombstones that were previously present for that row. This process is efficient because all of these tables are sorted.
- **==SSTable Indexing==**: Cassandra stores files that point to byte offsets in SSTable files to enable faster retrieval of data on-disk.
	- For example Cassandra might map a key of 12 to a byte offset of 984, meaning that data for key 12 is found at that offset in the SSTable. This is somewhat similar to how a B-Tree might point to data on disk.

### Gossip
- Cassandra nodes communicate information throughout the cluster via [[Gossip]], which is a peer-to-peer scheme for distributing information between nodes.
- Universal knowledge of the cluster makes every node aware and available to participate in all operations of the database, **eliminating any single points of failure** and allowing Cassandra to be a **very reliable database for availability-skewing system designs**. 
- Nodes track various information, such as **what nodes are alive/accessible**/**what the schema is**/etc.
	- For each node they know about, they manage:
		- `generation`: A timestamp when the node was bootstrapped
		- `version`: A [[Logical Clock]] value that increments every ~second.
	- Across the cluster, these values form a [[Vector Clock]], which allows nodes to ignore old cluster state information when it's received via Gossip.

Cassandra nodes routinely pick other nodes to gossip with, with a probabilistic bias towards "**==seed nodes==**", which are designated by Cassandra to boostrap the cluster and serve as guaranteed "hotspots" for Gossip, so that all nodes are communicating across the cluster.
- By creating these "choke points", Cassandra eliminates the possibility that sub-clusters of nodes emerge because information happens to not reach the entire cluster. 
- Cassandra ensures that Seed Nodes are always discoverable via off-the-shelf [[Service Discovery]] mechanisms.

### Fault Tolerance
- In a distributed system like Cassandra, nodes fail, and Cassandra must efficiently detect and handle failures to ensure that the database can read and write data efficiently: How is it able to achieve these requirements at scale?
- Cassandra uses a [[Phi-Accrual Failure Detector]] technique to detect failure during gossip; each node independently makes a decision on whether a node is available or not. 
	- When a node gossips with a node that doesn't respond, Cassandra's failure detection logic "convicts" that node and stops routing to it.
	- The convicted node can re-enter the cluster when it starts heartbeating again.
	- ==Cassandra will never consider a node truly "down" unless the Cassandra sysadmin decommisions the node or rebuilds it.== This is done to prevent intermittent communication failures or node restarts from causing the cluster to re-balance data.

In the presence of write attempts to nodes that are considered "offline", Cassandra leverages a technique called [[Hinted Handoff]]s
- When a node is considered offline by a coordinator node that attempts to write to it, the coordinator temporarily stores the write data in order for the write to proceed.
- This temporary data is called a ==hint==. When the offline node is detected as online, the node (or nodes) with a hint send that data to the previously-offline node.
![[Pasted image 20250521193853.png]]
Of note, a hinted handoff is mostly used as a **short term** way to prevent a node that is offline from losing writes. Any node that's offline for a long time will either be rebuilt or undergo [[Read Repair]]s, as hints usually have a short lifespan.

### How to use Cassandra

Data Modeling
- When leveraging Cassandra in system design, modeling your data to take advantage of its architecture and strengths is very important!
- If you come from a relational DB world, Cassandra data modeling might feel a little odd at first
	- While relational data modeling focuses on highly-normalized data, where you have a copy of each *entity* instance and you manage *relationships* between these entities via foreign keys and JOIN tables...
	- Cassandra is contrast **==doesn't have a concept of foreign keys/referential integrity/JOINs/etc.==** Cassandra also **doesn't favor normalization of data.** Instead, data modeling for Cassandra is ==**query-driven!**==
- Cassandra's data efficiency is heavily tied to the way that data is stored!
	- ==Cassandra lacks the query flexibility of relational databases (doesn't supports JOINS, services single-table queries==
	- ==So when we consider how to model the data of a Cassandra database, the [[Access Pattern]]s of the application must be considered first and foremost!==
		- It's important to understand what data is needed in each table, so that data can be denormalized (duplicated) across tables as necessary!
- The mains areas to consider are:
	- **==Partition Key:==** What data determines the partition that the data is on?
	- **==Partition Size:==** How big a partition is in the most extreme case, whether partitions have the capacity to grow indefinitely, etc.
	- **==Clustering Key:==** How the data should be sorted (if at all)
	- **==Data Denormalization:**== Whether certain data needs to be denormalized across tables to support the app's queries.

#### Example: Discord Messages
- Discord has a good [summary](https://discord.com/blog/how-discord-stores-billions-of-messages) of their use of Cassandra to store messages via blog posts, and it's a good model for how one might approach message storage for chat apps generally.
- **Use Case:** Discord channels can be quite busy with messages. Users typically query recent data given the fact that a channel is basically a big group chat Users might query recent data and scroll a bit, so having data sorted in reverse chronological order makes sense.
To service these needs, Discord originally created a `messages` table with the following schema:
```sql
CREATE TABLE messages (
  channel_id bigint,
  message_id bigint,
  author_id bigint,
  content text,
  PRIMARY KEY (channel_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
```
Above:
- You might wonder why **message_id** is used instead of a timestamp column like **created_at** for the sorting; Discord opted to eliminate the possibility of Cassandra primary key conflicts by assigning messages **Snowflake IDs**, which are basically chronologically sortable UUIDs. This is better than created_at because a Snowflake ID collision is impossible (it's a UUID), whereas a timestamp, even with millisecond granularity, has a likelihood of collision.
- I think is CQL that the above information is having a PK of only channel_id, with a clustering key of message_id. So a channel's messages are all on the same partition, and messages are sorted by chronlogical IDs, desc (basically by recency, desc). This is the order that we'd expect to have in a discord server!
- ==The above schema enables Cassandra to service messages for a channel via a **single partition**==
	- The partition key, `channel_id`, ensures that a single partition is responsible for servicing the query, preventing the need to do a [[Scatter-Gather]] operation across several nodes to get message data for a channel, which could be a slow/resource intensive.

**But even the above solution didn't fit all of Discord's needs!** 😱
- Some Discord channels can have an extremely high volume of messages. With the above schema Cassandra would struggle to handle large partitions for extremely busy Discord channels.
- **Large partitions** in Cassandra typically hit performance problems, and this was exactly what Discord observed.
- Additionally, Discord channels can perpetually grow in size with message activity, and would eventually hit performance problems if they live long enough. 

To solve this problem, Discord introduced the concept of a `bucket` and added it to the **Partition Key** part of a Cassandra primary key. A `bucket` represents 10 days of data, defined by a fixed window aligned to Discord's self-defined DISCORD_EPOCH of Jan 1, 2015.
The messages of even the most busy ==Discord channels over 10 days would certainly fit in a Cassandra partition==!
This also solved the issues of partitions growing monotonically; over time, a new partition would be introduced because a new `bucket` would be created!
And since people are really mostly reading new messages, ==Discord could query a single partition to service writes most of the time, because the most recent messages of a channel would usually be in one bucket! ==
- The only time they weren't was if:
	- A new bucket was created based on time passing
	- For inactive Discords, which were the significant minority of queries to the messages Cassandra table.

The revised schema:
```sql
CREATE TABLE messages (
  channel_id bigint,
  bucket int,
  message_id bigint,
  author_id bigint,
  content text,
  PRIMARY KEY ((channel_id, bucket), message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
```
- Above:
	- See that now we have a bucket column, with a composite partition key of (channel_id, bucket), and a clustering key of message_id (DESC)

**==See that Discord uses its channel access patterns to dictate its schema design, a great example of *query-driven data modeling.*==**
- Their choice of their primary key, including both partition key and clustering key components, is strongly linked to how data is accessed for their app.
- Finally, they had to think about partition size when designing the schema.


#### Example: TicketMaster
- Let's consider another use case: **TicketMaster's** ticket-browsing UI!
	- This is the UI that shows an event venue's available seats, and allows a user to select seats and then enter a seat checkout and payment flow.
- The TicketMaster **ticket browsing** UI is a UI that **doesn't need strict consistency**.
	- Event ticket availability changes, even as a  user is viwing the UI. Once a seat is selected and a purchase flow is attempted, the sytem can check a consistent database t odetermine if the seat is ACTUALLY available. Additionally, always showing the browsing UI is important, as a majority of users will browse, but a minority of users will actually *enter* a checkout flow!

When considering how to model our data to support a ticket browsing UI, we might consider every seat in an event a "ticket." 
- If we think about the access patterns of our system, we uncover that ==users will query data for a single event at a time==, and ==want to see totals of seats available== and ==also the seats themselves==!
- Users ==don't care about seeing the seats in any order==, since they will have an event venue map that dictates how they see seat availability.

Our first iteration of the schema might look like:
```sql
CREATE TABLE tickets (
  event_id bigint,
  seat_id bigint,
  price bigint,
  -- seat_id is added as a clustering key to ensure primary key uniqueness; order
  -- doesn't matter for the app access patterns
  PRIMARY KEY (event_id, seat_id)
);
```
- Above: It looks like **event_id** is the **Partition Key** and seat_id is implied to be the **Clustering Key** (e.g. we don't have to do a WITH CLUSTER ORDER BY (seat_id) phrase like we did in the past one. Perhaps the default is just sorting it ascending.)
- With the above schema, the app can query a single partition to service queries about the event, given that the primary key has a partition key of event_id.
	- The app can query the partition for price information about the event, for ticket availability totals, etc.
==**This schema has problems, however!**==
- For events with 10k+ tickets, the database needs to perform work to summarize information based on the user's query (price total, ticket total).
- This work might be performed A LOT for events that are very popular and have users frequently entering the ticket browsing UI.
![[Pasted image 20250521203842.png]]
Consider the above image: When a user clicks into a section of interest, the TicketMaster UI then shows the individual seats and ticket information!
![[Pasted image 20250521204137.png]]
This UX unveisls that we can add the idea of ==section_id== to our ticket table, and have the section id as part of the partition key!
- Then our tickets table now services the query to view individual seats for a given section.

The new schema looks like:
```sql
CREATE TABLE tickets (
  event_id bigint,
  section_id bigint,
  seat_id bigint,
  price bigint,
  PRIMARY KEY ((event_id, section_id), seat_id)
);
```
- The above schema is an improvement our original schema.
- This ==schema distributes an event over several nodes in the Cassandra cluster, because each section of an event is in a different partition.==
	- It also means that each partition is responsible for serving less data, because the number of tickets in a partition is lower.
	- This schema better maps to the data needs/access patterns of the TicketMaster ticket browsing UI.
- Q: But how how do we show ticket data for the entire event?
	- A: For this, we can consider a separate table called ==event_sections==!
```sql
CREATE TABLE event_sections (
  event_id bigint,
  section_id bigint,
  num_tickets bigint,
  price_floor bigint,
  -- section_id is added as a clustering key to ensure primary key uniqueness; order
  -- doesn't matter for the app access patterns
  PRIMARY KEY (event_id, section_id)
);
```
- The above table represents the idea of "==denormalizing==" data in Cassandra!
	- ==Rather than having our database do an aggregation== on a table or query multiple tables/partitions to service an app, ==it's preferable to denormalize information== like ticket numbers and a price floor in a section to make the access pattern for the app efficient!
		- The section states being queried don't need to be extremely precise; there's tolerance for [[Eventual Consistency]].
			- TicketMaster doesn't even show exact ticket numbers in their UI -- they merely show a total like **100+**
- The above table is partitioned by **event_id**. Cassandra will be responsible for querying many sections in one query, but events have a low number of sections (usually < 100), and this query will be served off a single partition.
- ==This means that Cassandra can efficiently query data to show the top-level venue view.==
	- ((I guess we somehow need to make sure that this is somewhat updated, yes?))


### Cassandra Advanced Feautres
- **==Storage Attached Indexes (SAI)==**: SAIs are a newer feature in Cassandra that offer global **secondary indexes** on columns. This offers flexible querying of data with performance that's worse than traditional querying based off partition key, but still good.
	- Enable users to avoid excess denormalization of data if there's query patterns that are less frequent.
- **==Materialized Views==**: A [[Materialized View]] is a way for a user to configure Cassandra to materialize tables based off a source table. They have some overlap with SQL views, except they actually "materialize" a table... As a User, this is convenient, because ==you can get Cassandra to denormalize data automatically for you!== This cuts complexity at your application level, since you don't need to author your application to write to multiple tables if data that is denormalized changes. 
- **==Search Indexing==**: Cassandra can be wired up to a distributed search engine like [[ElasticSearch]] or Apache [[Solr]] via different plugins; One example is the Stratio [[Lucene]] Index.


### Cassandra in an Interview
- When to use it:
	- Cassandra can be an ==awesome choice for systems that play to its strengths!==
	- ==A great choice in systems that prioritize [[Availability]] over [[Consistency]]== and have high scalability needs.
	- Can perform ==fast writes and reads at scale==, but is especially good for systems with ==high write throughput==, given its write-optimized storage layer based on [[LSM Tree]] indexing!
	- Cassandra' [[Wide-Column]] design make it a ==great choice as a database for flexible schemas or schemas that involve many columns that might be sparse==.
	- Cassandra is good when you have ==several clear access patterns for an application or use-case that can drive the design of your schema.==

- It's important to **know Cassandra's limitation!**
	- Cassandra isn't a great database choice for every system.
	- Cassandra ==isn't good for designs that prioritize strict consistency==, given its heavy bias towards availability.
	- It's important to adopt a query-driven data modeling approach to maximize the value Cassandra delivers in terms of write/speeds and scalability.



## Parting Thoughts:
From Stefan Mai, re: "Do I have to remember all of this?:
> Consider this general learning; it will be helpful for you as a SWE regardless!
> "From an interview-strategic standpoint, I'd recommend choosing a general data store you're comfortable with (Postgres, Redis, Cassandra, Dynamo) and then figuring out how/when you can apply it to various problems. For many problems it's ambiguous or stylistic which backend data store is "best" - by being familiar enough with one you'll be able to make a decent case for it."
