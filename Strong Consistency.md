---
aliases:
  - Linearizability
---
==All reads reflect the most recent write. All readers have the same view of the system. Every operation appears to happen atomically in one instant, respecting real-time order. Once a read completes, all other later reads see it.==  The system behaves as like one up-to-date copy of the data.
- Every operation takes effect *at one instant* between its start and finish, and real-time order is preserved.


It is achieved by [[Single-Leader]] (Leader-Follower) systems, consensus protocols like [[Raft]]/[[Paxos]], using [[Quorum Read]]s + [[Quorum Write]]s, and [[Synchronous Replication]].

> Ask: "Does the whole system have to behave like one up-to-date copy?"

Note: Many systems use this in a certain place where needed, but not throughout the system.


# How is it achieved?
- All reads and writes must coordinate through an authority that knows the latest committed value.
- Commonly this means a [[Single-Leader]] accepts writes and serves reads, or otherwise a [[Distributed Consensus|Consensus]] group like Paxos/Raft agrees on each operation order before it commits.
- Replicas do not independently answer reads unless they can prove that they are up to date, often through [[Quorum Read]]s or leader leases.

# Tradeoffs
- Has the highest coordination cost. This is the most expensive consistency model in terms of performance, but is necessary for systems that require absolute accuracy.
- . Higher [[Latency]], lower [[Availability]] during partitions, and harder to do cross-region.


# Use Cases
Situations where you need one globally-agreed truth. If two people try to take an item, only one should succeed. If a username is taken, every later check should see it as taken.
- Bank account balances
- Systems like TicketMaster (e.g. reserving a ticket, or buying a pair of a limited sneakers), inventory reservation
- [[Distributed Lock]]s
- Unique username creation


