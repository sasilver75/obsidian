---
aliases:
  - Consistent
  - Consistency Model
---
> Choose consistency by workflow, not by product (i.e. at a more granular level).
> Use linearizability when wrong ordering causes damage, or when stale reads cannot be tolerated.
> Use read-your-writes or monotonic consistency when the main risk is user confusion ("Where's my post?")
> Use causal consistency when event relationships matter.
> Use consistent prefix when ordered history matters.
> Use eventual consistency when availability/speed is paramount, and when staleness can be tolerated.


Options:
- [[Strong Consistency]]/[[Strong Consistency|Linearizability]]
	- System behaves as if there is a single copy of the data; every operation takes effect at one instant between its start/finish, and real time order is preserved.
	- Mechanism: Globally coordinate operation order before answering.
	- [[Strong Read-After-Write Consistency]] ((Can Pretty Much Ignore This One, sort of a weird one))
		- Once a write is acknowledge, later reads see that write, but this guarantee is usually scoped to an object, partition, region, or API guarantee.
		- Mechanism: Make reads wait for or route to a replica containing the acknowledged write.
		- Comparison with Strong Consistency:  Weaker than SC only when its scope is narrower, such as guaranteeing only that the writer sees the completed write. If it means that every client's later read of the same object must see every completed write, then for that object it is essentially the same as SC for ordinary reads and writes. SC is the broader term because it defines the whole ordering model, including what happens with concurrent reads, concurrent writes, and multi-object state.
- [[Sequential Consistency]]
	- All clients observe one shared sequential order of operations, and each client's own operation order is preserved.
	- Weaker than linearizability, because the shared order doesn't need to respect real-time order between different clients.
	- Mechanism: Maintain a single logical operation order, but allow that order to lag or differ from wall-clock completion order.
- [[Causal Consistency]]
	- If one write causally depends on another (e.g. Friend Request Sent, Friend Request Accepted), everyone must observe them in that order. Independent concurrent writes may be seen in different orders.
	- Mechanism: Track dependencies and reveal dependent writes only after their causes.
- Session Guarantees 
	- [[Read-your-Writes Consistency]]
		- A client always sees its own writes, even if other clients may temporarily see older data.
		- Mechanism: Track each client's last write version and enforce that on later reads.
	- [[Monotonic Reads Consistency]]
		- Once a client has seen a variation of data, future reads by that client will not return older versions.
		- Mechanism: Track each client's highest observed version and never serve older data.
- [[Consistent Prefix Consistency]]
	- Reads see a prefix of the write history. You might see stale data, but you will not see later writes without the earlier writes that came before them.
	- Mechanism: Apply an ordered log in order and read from a known prefix.
	- Difference from Causal Consistency: CP preserves order in some write history/log/shard/partition, while CC preserves cause-and-effect relationships even across objects or partitions when dependencies are track.
- [[Eventual Consistency]]
	- If writes stop, all replicas *eventually* converge to the same value. Before convergence, reads may return different or stale values.
	- Mechanism: Replicate asynchronously and reconcile conflicts later via [[Conflict Resolution]] strategies.


Session Guarantees


Rule of Thumb:
- Use [[Strong Consistency|Linearizability]] when wrong ordering causes real damage
- Use [[Read-your-Writes Consistency]] or [[Monotonic Reads Consistency]] when the main problem is user confusion.
- Use [[Causal Consistency]] when relationships between events matter.
- Use [[Consistent Prefix Consistency]] when ordered history matters.
- Use [[Eventual Consistency]] when freshness is less important than speed, availability, scale.