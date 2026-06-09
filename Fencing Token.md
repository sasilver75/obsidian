
A monotonically increasing value issued to clients when it acquires a lock, lease, or leadership role in a distributed system. Its purpose is to prevent an old or paused client from performing writes after it has lost ownership.
- See [[Distributed Lock]] for a description of the Stale Lease problem, and how fencing tokens help address it.


When a client gets a lock, it receives a token:
- Client A acquires lock, and receives `token 41`
- Client B later acquires a lock and receives `token 42`

> Any downstream system that accepts writes stores the highest fencing token it has seen for that resource. It rejects operations with a token lower than that stored value.

So in a scenario where:
```
T=0   Worker A gets lock for 30s with token 41
T=5   Worker A freezes (e.g. garbage collection pause)
T=30  Lock expires
T=31  Worker B gets lock with token 42
T=32  Worker B writes correct value
T=35  Worker A wakes up and writes old value
	... Worker A's write is rejected! 🚨
```
In this scenario, ==The protected resource only accepts writes with a token newer than the last accepted token.== Obviously, this stops the Stale Owner problem from corrupting state after their lease expires!






