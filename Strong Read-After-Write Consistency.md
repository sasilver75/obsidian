---
aliases:
  - Session Read-After-Write Consistency
  - Per-Object Read-After-Write Consistency
  - Regional Read-After-Write Consistency
  - Query Read-After-Write Consistency
---


A [[Consistency]] model.

==After a write is acknowledges, subsequent reads see that write immediately.==  If global, this is close to [[Strong Consistency|Linearizability]] for that object/key.
- This is ==usually scoped to an object, partition, region.==


# How is it achieved?
- The system acknowledges  a write only after it is durable somewhere authoritative, then routes later reads to a place that has that write.
- This can mean reading from the leader, waiting for replicas to apply up to the write's log position, using [[Quorum Write]]s plus [[Quorum Read]]s, or attaching a version/timestamp to the write and requiring reads to wait until a replica has reached that version.

# Tradeoffs?
- Reads may need to wait for replication.
- Read routing becomes more constrained. Cheaper than full linearizability if scoped narrowly, but still adds coordination.

# Use Cases
- Uploading a file, publishing an article, creating a record and then immediately loading it by ID
- After you `POST /file/123`, a later direct read like `GET /file/123` should return the new object, but lists, search indexes, analytics, or other derived views might still lag.

# Often the guarantee is *scoped* in one of these these ways
- Session read-after-write
	- After *I* write x=1, *my* later reads see x=1. Other clients might still see x=0.
- Per-Object read-after-write
	- After x=1 completes, later reads of x see 1. This says nothing about where a multi-object read of x and y is one coherent snapshot.
- Query read-after-write
	- Sometimes direct lookup is fresh, but search/list/count/query results lag. `GET /item/123` sees the write, but `GET /items?tag=fooo` might not include it.
- Regional read-after-write
	- Reads in the write's region see it, but cross-region reads may lag.



#  How is this different from [[Strong Consistency]]?
- Linearizability is a stronger guarantee. It guarantees a single real-time order for ALL CLIENTS' OPERATIONS! The whole system behaves like a single up-to-date copy, applying to all clients.
- Strong read-after-write consistency usually guarantees only this narrower thing:
	- *"After a write is acknowledged, later reads see that write."* This maybe scoped to one key, one object, one partition, one region, or sometimes one client/session.
	- A completed write becomes visible to later reads, but usually only within a defined scope: same client, same object, same partition, same region, or same read path.

Example:
```
Alice writes x=1
Write completes
Box reads x
```
Above:
- Under [[Strong Consistency]]:
	- Bob must see x=1 if his read starts after Alice's write completed. 
- Under [[Strong Read-After-Write Consistency]]:
	- Bob may or may not be guaranteed to see x=1, depending on the scope of the guarantee.
		- If it is *global read-after-write*, bob should see it.
		- If it is *session read-after-write*, only Alice is guaranteed to see it.
- Difference is order: Linearizability preserves real-time order across all operations. Strong read-after-write does not necessarily define a single global order for every read/write interaction.

Example with two keys:
```
Alice writes x=1
Bob writes y=1
Carol reads x=1, y=0
Dave reads x=0, y=1
```
This *can be* compatible with weaker read-after-write guarantees.