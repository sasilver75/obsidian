
A failure mode in [[Lock]]s/[[Distributed Lock]]s where a client once held a valid time-limited lease, the lease expires or is replaced, but the client later continues acting as if its lease is still valid.

A lease is a lock with an expiration time, granting temporary permissions to perform some work until time T.

The stale lease problem happens when reality changes, but Client A doesn't observe that change in time correctly:
```
t=0   Client A acquires lease for 10 seconds.
t=1   Client A starts updating a shared resource.
t=2   Client A is paused by garbage collection, scheduler delay, VM suspend, or network trouble.
t=10  Client A's lease expires.
t=11  Client B acquires the same lease.
t=12  Client B writes the correct new state.
t=30  Client A resumes and writes old state, still believing it owns the lease.
```
Process delays/pauses can be caused by garbage collection, OS scheduling delays, VM suspensions, network partitions, slow I/O, overloaded machines, etc.

This is typically solved using [[Fencing Token]]s, where recipients of a lease receive a monotonically increasing integer (e.g. `42`) that they send along with their (e.g.) write to the resource they acquired the lease for, and that protected resource denies writes with fencing tokens lower than the highest one they've seen.

```
A lease says: "You may act for now."
A fencing token lets the resource say: "You are older than the current lease holder, so I reject you."
```
A stale lease problem exists whenever an expired or superseded lease holder can still perform side effects after losing the lease.







