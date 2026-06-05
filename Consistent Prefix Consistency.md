A [[Consistency|Consistency Model]]

Reads see a *prefix* of the write history. You might see stale data, but you will never see later writes without the earlier writes that came before them.
- Often scoped; We could talk about a single global consistent prefix, or a per-partition consistent prefix, etc.

# How is it achieved?
- Writes are placed into an *ordered [[Log]]*. 
- Replicas apply that log in order.
- A read is served from some replica or snapshot who is at some particular *log offset,* so it sees all writes up to that point and none after it.
- This is common in [[Single-Leader]]/[[Single-Leader|Leader-Follower]] replication where followers replay the leader's log sequentially (([[Asynchronous Replication]])).

# Tradeoffs
- Avoids out-of-order states, but reads can still be stale. 
- It does not guarantee (e.g.) [[Read-your-Writes Consistency]] unless combined with session tracking.

# Use Cases
- Event logs, order history, replicated database followers, audit streams
- Readers may be behind, but they should see history in order.
- An order status page can show `created -> paid` while missing the later `shipped` event, but it should never show `shipped` without first showing `paid`.


# Comparison with [[Causal Consistency]]
Q: If you don't see later writes without earlier writes, does this imply [[Causal Consistency]], which afaik is a more stringent consistency model?
- Yes, but only if the consistent prefix is over one global write history that respects causality. 
- In practice, the difference is usually **scope**
	- Consistent prefix often means: *"For a given log/shard/partition/replica stream, you see writes in order*", but it does not necessarily mean *Across all objects/shards/users/regions, causal dependencies are tracked/enforced."*
- So...
	- Consistent Prefix: Preserves order within some write history/log
	- Causal Consistency: Preserves cause-and-effect relationships, even across objects/partitions, if the system tracks them.
- ==These models are not always a perfect strict ladder.== A single global consistent prefi can be stronger than some forms of causal consistency, but per-partition consistent prefix is weaker because it may miss cross-partition causal dependencices.
