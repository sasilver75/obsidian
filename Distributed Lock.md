

Distributed locks are a way to make sure that only one worker/process/node performs some action at a time across multiple machines, protecting a shared resource across many processes.

Common use cases:
1. Singleton background jobs: Only one server should run the hourly billing job, or rebuild a search index.
2. Preventing duplicate work: Multiple workers may notice the same task; a lock prevents all from processing it at once.
3. Coordinate access to shared resources: Updating a shared file, running a migration, allocating scarce inventory.
4. [[Leader Election]]: One node becomes the leader and performs coordination.
5. Rate-limited external systems: Only one process talks to a fragile third-party API for a given customer account/resource, e.g. [[Request Coalescing]] in the case of a [[Cache Stampede]]

==Locks are often OVERUSED! In many cases, it's better to use any of the following:==
- Database transactions
- Uniqueness constraints
- [[Idempotency|Idempotency Key]]s
- [[Message Queue]]s (which enforce linearity)
- [[Optimistic Concurrency Control]] (OCC)
- [[Compare-and-Swap]] updates
- [[Leader Election]] primitives
- [[Event Sourcing]] and [[Transactional Outbox Pattern|Outbox Pattern]]s


# Why Distributed Locks are Hard

For a simple lock:
```
acquire lock
do work
release lock
```

But distributed systems add failure modes:
- The lock holder crashes
- The network has a partition
- The process freezes temporarily during a [[Garbage Collection|GC]] pause.
- The lock expires while the worker is still working.
- Two nodes disagree about who owns the lock.
- A release request from an old owner deletes a new owner's lock.
- Clocks drift.
- The lock service itself fails.

So real distributed locks are usually ==leases== ("You own this lock for 30 seconds unless you renew it"), not permanent locks, so that if the owner dies, the lock eventually becomes available again.

==Rule of thumb:== For efficiency-only, Redis-based locking is usually fine. For business correctness, prefer DB constraints/transactions if possible, else use etcd/ZooKeeper/Consul with fencing tokens.
> “I’d first ask whether this needs correctness or just duplicate-work reduction. If correctness matters, I’d prefer database constraints or a consensus-backed lease with fencing tokens. If it’s only to reduce redundant work, a Redis SET NX PX lock with owner IDs, safe release, TTL, retry/backoff, and observability is probably enough.”


# Basic [[Redis]] Implementation
- A common simple implementation uses Redis:
```
SET lock:customer:123 <unique-owner-id> NX PX 30000
```
Meaning:
- `NX`: Only set if the key doesn't already exist
- `PX 30000`: Expire after 30 seconds

If redis returns `OK`, you've got the lock!

To release it safely, you must only delete it if you still own it! Do not blindly call `DEL lock:key`,  because your lock might have expired and someone else might own it. Instead, this is usually done with a [[Lua]] script:
```lua
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
  else
    return 0
  end
```
(Redis doesn't have a native single command for "delete if value equals X," so people use a Lua script, which Redis can execute atomically.)

==This Redis style is fine for "avoid duplicate work" use cases, but is less ideal for correctness-critical oprations like money movement, inventory guarantees, or exactly-once workflows.==

### The Stale Owner Problem
- This is the most important concept! Imagine:
```
T=0   Worker A gets lock for 30s
T=5   Worker A freezes (e.g. garbage collection pause)
T=30  Lock expires
T=31  Worker B gets lock
T=32  Worker B writes correct value
T=35  Worker A wakes up and writes old value
```
==Now both owners acted as if they owned the lock!==
This is why a [[Time to Live|TTL]] alone is not enough for correctness!

### [[Fencing Token]]s
- A stronger pattern uses Fencing Tokens: a ==monotonically-increasing number issued with each lock acquisition.==
```
Worker A gets lock, token = 41
Worker B gets lock, token = 42
```
==The protected resource only accepts writes with a token newer than the last accepted token.== Obviously, this stops the Stale Owner problem from corrupting state after their lease expires!

(Redis can issue fencing tokens, but the resource being protected must enforce them atomically. If the resource can't do atomic conditional writes, fencing token don't give you correctness. This means that fencing works well with Postgres/MySQL conditional updates, DynamoDB conditional writes, etcd revisions, etc.)

# [[Distributed Consensus|Consensus]]-based Locks
- Stronger distributed locks are usually built on consensus systems like [[Apache ZooKeeper|ZooKeeper]], [[etcd]], or [[HashiCorp Consul|Consul]], which use protocols like [[Raft]] or [[Zookeeper Atomic Broadcast|ZAB]] to agree on state.

A typical ZooKeeper-style lock works something like:
1. Client creates an ephemeral sequential node:
```
/locks/payment/lock-000001
/locks/payment/lock-000002
/locks/payment/lock-000003
```
2. The client with the smallest sequence number owns the lock
3. Other clients watch the node before them
4. If the owner dies, its ephemeral node disappears
5. The next client becomes owner

This gives better ordering and failure handling than a single Redis instance.


# Database-Based Locks
- Databases themselves can also implement distributed locking:
```sql
INSERT INTO distributed_locks (name, owner_id, expires_at)
VALUES ('billing-job', 'worker-123', now() + interval '30 seconds');
```
Above: as long as `name` has a uniqueness constrain, only one insert succeeds!

To acquire an expired lock:
```sql
UPDATE distributed_locks
SET owner_id = 'worker-456',
  expires_at = now() + interval '30 seconds'
WHERE name = 'billing-job'
AND expires_at < now()
```

If the resource you're protecting is already in the same database, though, DB-native locking will just be better.