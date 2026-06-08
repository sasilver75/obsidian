A resilience pattern that *isolates parts of a system*, so that one failure cannot consume all shared resources.
- The reference comes from ships, where bulkheads divide a ship into watertight compartments, so that damage in one compartment doesn't sink the whole ship.

The key idea: ==failure should be contained to the smallest reasonable area, instead of spreading through shared resources.==

# Example 1: Threads
Imagine that your application has 10 worker threads in total, talking to (Payments, Search, Recommendations). Without bulkheads, all 10 workers are shared. If Recommendations becomes slow, it might occupy all 10 workers. Then Payments and Search cannot run, even though they are healthy.

With a bulkhead pattern, you might split the workers:
- Payments: 4 workers
- Search: 4 workers
- Recommendations: 2 workers

Now if Recommendations becomes slow, it can only block its 2 workers!


# Example 2: Database Connection Pools
A database can only handle so much concurrent work: connections, CPU, locks, memory, I/O, query slots, etc. A bulkhead protects important database traffic from less important or riskier traffic.

Imagine that we've got two kinds database work:
1. A Web API serving user-facing requests, like loading account pages
2. A reports worker running background analytics jobs, like generating reports

Let's say that our database has a maximum useful capacity of around 40 connections.

Without a bulkhead, both workloads might be configured too generously:
- Web API pool: Max 30 connections
- Report Worker pool: Max 30 connections
- Database capacity: ~40 connections

If reporting gets busy, it might open and hold many connections! User requests them compete with reporting queries for database capacity. The web app might become slow or fail, even though the user-facing code is healthy.

With a bulkhead, you can reserve capacity by workload:
```
Web API pool:       max 35 connections
Report Worker pool: max 5 connections
Database capacity:  ~40 connections
```
Now reporting can still run, but it cannot consume the whole database. If reports get slow or pile up, they're still limited to their 5 connections. The Web API still has most of the capacity available.



_______

Q: How do you actually size a database capacity in actuality?

A: You're usually not given a formula like "this database supports 40 connections."

You find two different numbers:
1. Hard connection limit: How many client sessions the database can technically accept.
2. Useful concurrency limit: How many queries can run at once before latency/locks/CPU/memory/IO get bad.

For bulkheads, you care mostly about the second one.

For this, you want to:
1. Measure the workload 
	- Request rate, queries per request, average+p95 query time, how long each request holds a connection, read/write mix, lock contention, CPU/memory/IO/network usage
2. Estimate active DB concurrency
	- Little's Law: A useful rule of thumb: `active DB concurrency ~= requests per second * time holding DB connection`. So 400 requests/sec with 50ms requests -> 400*.050=20 active connections, so a pool of ~20-30 might be plenty.
3. Load test to find the knee
	- Increase concurrency gradually and watch latency. Increase the number of active connections, looking for the point where acceptable latency turns into p95/p99 latency jumps.
4. Leave headroom
	- Don't just allocate 100$ of observed capacity; reserve room for traffic spikes, admin sessions, migrations, [[Failover]]s, [[Vacuum]], etc. 
	- If the "knee/elbow" testing showed that at around 50 DB connections, you usually set normal operating limits below that, maybe 35-40, so that you have headroom.
5. Allocate pool sizes across clients
	- If we've determined that useful DB capacity is 40 active connections, we might allocate them thusly:
		- Web API: 30
		- Report worker: 5
		- Admin/internal: 3
		- Reserve: 2

Key idea: Size your pools around **measured useful concurrency,*** not around the database's theoretical or billed max concurrency setting. More connections doesn't automatically mean more throughput. Past a certain point, they just create more waiting inside the database, instead of at the application boundary. This can cause [[Cascading Failure]]s, etc.




