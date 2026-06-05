---
aliases:
  - Consistent
  - Consistency Model
---



Options:
- [[Strong Consistency]]/[[Strong Consistency|Linearizability]]
	- System behaves as if there is a single copy of the data; every operation takes effect at one instant between its start/finish, and real time order is preserved.
	- Mechanism: Globally coordinate operation order before answering.
- [[Strong Read-After-Write Consistency]]
	- Once a write is acknowledge, later reads see that write, but this guarantee is usually scoped to an object, partition, region, or API guarantee.
	- Mechanism: Make reads wait for or route to a replica containing the acknowledged write.
- [[Read-your-Writes Consistency]]
	- A client always sees its own writes, even if other clients may temporarily see older data.
	- Track each client's last write version and enforce that on later reads.
- [[Monotonic Reads Consistency]]
	- Once a client has seen a variation of data, future reads by that client will not return older versions.
	- Track each client's highest observed version and never serve older data.
- [[Causal Consistency]]
	- If one write causally depends on another (e.g. Friend Request Sent, Friend Request Accepted), everyone must observe them in that order. Independent concurrent writes may be seen in different orders.
	- Track dependencies and reveal dependent writes only after their causes.
- [[Consistent Prefix Consistency]]
	- Reads see a prefix of the write history. You might see stale data, but you will not see later writes without the earlier writes that came before them.
	- Apply an ordered log in order and read from a known prefix.
- [[Eventual Consistency]]
	- If writes stop, all replicas *eventually* converge to the same value. Before convergence, reads may return different or stale values.
	- Replicate asynchronously and reconcile conflicts later via [[Conflict Resolution]] strategies.


Rule of Thumb:
- Use [[Strong Consistency|Linearizability]] when wrong ordering causes real damage
- Use [[Read-your-Writes Consistency]] or [[Monotonic Reads Consistency]] when the main problem is user confusion.
- Use [[Causal Consistency]] when relationships between events matter.
- Use [[Consistent Prefix Consistency]] when ordered history matters.
- Use [[Eventual Consistency]] when freshness is less important than speed, availability, scale.