---
aliases:
  - Redlock
---
A distributed lease algorithm for coordinating work along multiple processes using several independent redis masters... tries to ensure that only one client holds a [[Distributed Lock]] at a time, even if some Redis nodes fail, by requiring the client to acquire the same lock on a majority of Redis nodes before the lock is considered valid.

Important caveat: Redlock is best understood as a fault-tolerant, time-limited lease, not as a perfect distributed mutex. For correctness-critical systems, you usually need stronger mechanisms such as DB [[Transaction]]s, [[Optimistic Concurrency Control|OCC]], [[Fencing Token]]s, or consensus-oriented stores like [[etcd]], [[Apache ZooKeeper|ZooKeeper]], or [[HashiCorp Consul|Consul]].


Imagine you have five application servers that are all trying to run the same job, but you only want one server to do the work.

A normal Redis lock using one Redis instance is simple:
```
SET lock:tenant_123 random_value NX PX 30000
```
`NX`: Set iff the key does not already exist.
`PX 30000`: Expire after 30,000 ms (30s)
`random_value`: A unique ownership token

==A single Redis instance is as single point of failure, though! If that Redis primary crashes before replicating the lock to a replica, another client may acquire the same lock after failover, while the first client still believes it owns the lock.==
- This is *different* (almost the inverse case) of the "[[Stale Lease]]" problem when a lease borrower crashes, comes back up, and thinks that it still owns the lease.

==Redlock tries to reduce this failure mode by using several independent Redis masters, instead of one Redis primary plus replicas.==
- Think of Redlock as asking several independent judges for permission; you don't need every judge to say yes, you need a majority.
- With (typically) five Redis masters (`A B C D E`), the client considers the lock acquired if any only if it successfully writes the lock to at least three of the five Redis nodes, and does so quickly enough that the lock still has useful time remaining.

To acquire a lock:
1. The client generates a unique random value
2. The client records the current time
3. The client tries to acquire the same lock on all Redis nodes, usually in parallel
4. Each Redis node succeeds only if the lock key does not already exist.
5. The client counts successful Redis responses.
6. The client considers the lock acquired only if both conditions are true:
	- successful_nodes >= majority  (3 nodes of 5)
	- elapsed_time < ttl_ms
7. The usable lock validity is shortened by the time spend acquiring the lock:
	- effective_validity = ttl_ms - elapsed_time - clock_drift_margin
8. If the client fails to acquire a majority, or if the acquisition took too long, the client releases the lock from every Redis node where it might have acquired it.
9. To release the lock, the client deletes the key only if the stored value matches its random value that it set.


Example:
```
TTL = 10 seconds
Redis nodes = 5
majority = 3
```
Worker A tries to acquire `lock:monthly-invoice-2026-06`
- Sets `SET NX PX 10000` to all five Redis nodes
```
Redis A: success
Redis B: success
Redis C: success
Redis D: timeout
Redis E: failure
```
Worker A got 3/5 in 80ms
Worker A thinks to itself
```
I own the lock for roughly 10,000ms - 80ms - drift margin.
```
- If Worker A takes 20 seconds, the lock may expire and Worker B may acquire the same lock. Worker A might still be running, but Worker A no longer has a valid lease.
- That is the core danger: Redlock does not magically stop a slow or paused process from continuing to act after its lease has expired.


Martin Kleppmann argued that Redlock is ==unsafe for correctness-critical locking== because real distributed systems can have long process pauses, network delays, and clock issues. His key point is that even if lock acquisition works, a client can pause for longer than the lease, resume, and then write stale data. This is the [[Stale Lease]] problem.
- The usual fix is a [[Fencing Token]], which is a monotonically increasing number issued with each lock acquisition:
```
Client A gets token 41
Client B later gets token 42
```
- The protected resource then rejects writes with older tokens.
- RedLock's random value is NOT a fencing token.
```
If duplicate work is merely inefficient, Redlock may be acceptable.
If duplicate work corrupts state, use fencing tokens or a stronger coordination mechanism.
```
- Use [[etcd]], [[Apache ZooKeeper|ZooKeeper]], or [[HashiCorp Consul|Consul]] if you need a real distributed coordination service.