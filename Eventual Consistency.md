---
aliases:
  - Eventually Consistent
---
A [[Consistency|Consistency Model]].

If writes stop, all replicas *eventually* converge to the same value, but before convergence, reads may return different or stale values. 
- This is the most relaxed form of consistency, often the default behavior of most distributed databases.
- Used with systems like [[Domain Name Service|DNS]] where temporary inconsistencies are acceptable and [[Availability]] is paramount.

# How is it achieved?
- Writes can be accepted by different replicas without immediate global coordination.
- [[Asynchronous Replication]]: Replicas asynchronously exchange updates through replication streams, [[Gossip]], background repair, anti-entropy, or read repair. 
- If replicas receive conflicting writes, the system resolves them using (e.g.) [[Last Write Wins]], merge functions, application conflict handling, or [[Conflict-Free Replicated Datatype|CRDT]]s

# Tradeoffs
- High availability and low latency, especially across regions, but clients may see stale reads, conflicting values, non-monotonic reads, or temporary disagreement.

# Use Cases
- Like counts, view counts, recommendations, search indexing, metrics, caches, DNS-style replication.
- Temporary disagreement is fine as long as replicas converge later.
- A post may show 101 likes in one region and 104 in another, but eventually they settle.