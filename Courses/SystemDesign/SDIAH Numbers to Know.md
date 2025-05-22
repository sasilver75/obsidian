Link:  https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know

Often times a book published just a few years ago will be teaching **patterns** that still make sense, but quoting numbers that are off by orders of magnitude.

One of the biggest giveaways that a candidate has book knowledge but no hands-on experience during a system design interview is that they rely on outdated hardware constraints.

They do scale calculations using numbers from 2020 that dramatically underestimate what modern systems can handle! This isn't the candidate's fault, but understanding modern hardware capabilities is crucial for making good system design decisions.
- When to shard a database
- Whether to cache aggressively
- How to handle large objects
All depend on having an accurate idea of what today's hardware can handle!

## Numbers that matter in 2025

Modern Servers
- An AWS M6i.32xlarge comes with
	- 512GiB memory
	- ==128 vCPUs== for general workloads
- Memory-optimized instances go further:
	- X1e.32xLarge provides 4TB or RAM
	- U-24tb1.metal provides ==24TB of RAM==
- This shift matters, because ==many applications that once required distributed systems can now run on a single machine!==

Storage Capacity 
- AWS i3en.24xlarge provides ==60TB of local SSD storage==.
- D3en.12xlarge offers ==336 TB of HDD storage== for data-heavy workloads!
- Object storage like S3 is effectively unlimited, handling petabyte-scale deployments easily.
- ==The days of storage being a primary constraint are largely behind us.==

Network Capabilities
- Within a datacenter, ==10Gbps is standard==, with higher performance instances supporting up to ==20Gbps==.
- Cross-region bandwidth typically raanges from ==100Mbps-1Gbps==.
- ==Latency remains predictable==: ==1-2 ms== within a region, and ==50-150ms cross-region==.

When textbooks talk about splitting databases at 100GB or avoiding large objects in memory, they're working form outdated constraints. The hardware running our systems today would have been unimaginable a decade ago, and these capabilities fundamentally change how we approach SD!


# Applying these numbers in SD interviews

### Caching
- In-Memory caches have grown exponentially in size and capability. Gone are the days of 32-64GB Redis instances; ==today, we routinely handle TB-scale datasets with single-digit millisecond latency, and a single instance can process hundreds of thousands of operations per second.==
**Numbers to know**:
- Memory: Up to ==1TB RAM== on memory-optimized instances, with some configurations exceeding this for specialized use cases.
- Latency: 
	- Reads: <1 ms within same region
	- Writes: 1-2 ms average cross-region for optimized systems
- Throughput:
	- ==Reads==: ==More than 100k rps per instance== for in-memory cache like ElastiCache Redis on modern Graviton-based nodes (Graviton is an AWS processor).
	- ==Writes==: sustained throughput of ==hundreds of thousands of requests per second==.
**When to consider sharding:**
- Dataset size: Approaching 1TB in size
- Throughput: Sustained throughput of > 100k ops/second
- Read Latency: Requirements below 0.5ms consistency (if being exceeded, consider sharding)

==These capabilities fundamentally change caching strategies!==
- The ability to **CACHE AND ENTIRE DATABASE IN MEMORY**, even at hundreds of gigabytes, means you can often avoid complex partial caching schemes together!
- When you DO need to scale, the bottleneck is usually operations per second or network bandwidth, NOT memory size -- a counterintuitive shift from a few years ago.

### Databases
- The raw power of modern databases surprises even experienced engineers.
- Single [[PostgresDB|Postgres]] or [[MySQL]] instances now routinely handle ==dozens of TB of data while maintaining millisecond-level response times, and handle tens of thousands of transactions per second on a single primary==, with the bottleneck often being operational concerns rather than performance limits.
**Numbers to Know**:
- ==Storage==: Single instances handle ==up to 64TiB== for most database engines, with [[AWS Aurora]] supporting up to ==128TiB== in some configurations.
- ==Latency==: 
	- Reads: ==1-5ms for cached data==, ==5-30ms for disk== (optimized configurations for RDS and Aurora)
	- Writes: ==5-15ms== for commit latency (single node, high-perf setups)
- ==Throughput==:
	- Reads: ==Up to 50k TPS== in a single node configuration on Aurora and RDS
	- Writes: ==10-20k TPS== in single-node configurations on Aurora and RDS
- ==Connections==: 
	- ==5-20k concurrent connections==, depending on database and instance type
**When to consider Sharding**:
- ==Dataset size==: Approaching 50TiB
- ==Write throughput==: Consistently exceeding 10k TPS
- ==Read Latency==: Requirements below 5ms for uncached data may necessitate optimization
- ==Geographic Distribution==: Cross-region replication or distribution needs
- ==Backup/Recovery==: Backup windows that stretch into hours or become operationally impractical.

**WARN:** Many candidates jump to distributed solutions too early. For systems handling millions or even tens of millions of users, a single well-tuned database can often handle the load.
- If you DO need to scale, consider what's driving the decision to better understand tradeoffs:
	- Pure data volume
	- Operational concerns like backup windows
	- The need for geographic distribution
- ==Often, candidates have 500GB or a few TB of data and candidates are mistakenly talking about how they'd shard the database; slow down, do the math, and make sure that sharding is actually NEEDED before you start explaining how you'd do it!==


## Application Servers
- Modern application servers have evolved beyond the resource constraints that shaped many traditional design patterns.
- Today's servers routinely handle ==thousands of concurrent connections with modern resource units==, and cloud platforms enable ==near-instant scaling in response to load!==

Numbers to know:
- Connections: ==100k+ concurrent connections== per instance for optimized configuartions
- CPU: ==8-64 Cores==
- Memory: ==64-512GB== standard, up to 2TB available for high-memory instances.
- Network: Up to ==25Gbps bandwidth== in modern server configuratinos
- Startup Time: ==30-60 seconds for containerized apps==

When to consider Sharding:
- ==CPU Utilization==: Consistently above 70-80%
- ==Response Latency==: Exceeding SLA or critical thresholds
- ==Memory Usage==: Trending above 70-80%
- ==Network Bandwidth==: Approaching 20 Gbps

While the trend towards stateless services are valuable for scaling, don't forget that each server does have substantial memory available!
 - ==Local caching==, in-memory computations, and session handling can all leverage this memory to improve performance dramatically!
 - PU is almost always your first bottleneck, not Memory, so don't shy away from memory-intensive optimizations when they make sense.

### Message Queues
- Have transformed from simple task delegation systems into high-performance data highways.
- Modern systems like [[Kafka]] can process ==millions of messages per second with single-digit millisecond latency==, while maintaining weeks or months of data!
	- This combination of **Speed** and **Durability** have expanded their role far beyond traditional async data processing. 
**Numbers to Know**:
- Throughput: Up to ==1M messages/second per broker== in modern configuraations
- Latency: ==1-5ms end-to-end== within a region for optimized setups
- Message Size: ==1KB-10MB== efficiently handled
- Storage: Up to ==50TB per broker== in advanced configurations
- Retention: ==Weeks to months of data==, depending on disk capacity and configuration.
When to consider sharidng:
- ==Throughput==: Nearing 800k messages/second per broker
- ==Partition Count==: Approaching 200k per cluster
- ==Consumer Lag==: Consistently growing, impacting real-time processing
- ==Cross-Region Replication==: IF geographic redundancy is required.


### Cheat Sheet
- here's a one-stop shop for numbers you might want to know in 2025:
![[Pasted image 20250520155119.png]]


### Common Mistakes Interviews

- ==**Premature Sharding**==
	- Candidates assuming that sharding is already necesssary; Introducing a data model and immediately explaining which column they'd shard on.
	- It comes up almost every time in Design Yelp in particular:
		- We have 10M businesses, each of which is roughly 1kb of data; this is 10M * 1kb = 10GB of data! If we 10x it to account for reviews which we can store in the same database, then we're only at 100GB -- why shard?
	- The same thing comes up a lot in Caches:
		- Take a LeetCode leaderboard where we have 100k competitions and up to 100k users per competition. We're looking at 100k x 100k x 36b ID + 4b float rating = 400GB
		- This can still fit on a single large cache -- no need to shard!
- ==**Overestimating Latency**==
	- I see this most often with SSDs.
	- Candidates tend to vastly overestimate the latency additional to query an SSD (Database) for a simple key or row lookup. We're talking ==10ms== or so -- it's fast!
	- ==Candidates will often try to justify adding a caching layer== to reduce latency when the simple row lookup is already fast enough -- ==no need to add additional infrastructure where it's not needed!==
- **==Overengineering given a high write throughput==**
	- Incorrect estimates routinely lead to over-engineering. 
	- ==Imagine that we have a system with 5,000 writes per second! Candidates will often jump to adding a message queue to buffer this "high" write throughput, but they don't need to!==
	- Let's put this in perspective:
		- **A well-tuned [[PostgresDB|Postgres]] instance with simple writes should be able to handle 20k+ writes per second!**
		- ==What actually limits write capacity are== things like complex transactions spanning multiple tables, [[Write Amplification]] from excessive indexes, writes that trigger expensive cascading updates, or heavy concurrent reads competing with writes.
	- Message Queues become valuable when you need **guaranteed delivery** in case of downstream failures, event sourcing patterns, or handling write **spikes** above 50k+ WPS, or just generally decoupling producers from consumers... but they add complexity and should be justified by actual requirements!
		- Before reaching for a message queue, we should consider simpler optimizations like **==batch writes==**, **optimizing our schema and indexes**, and **using ==connection pooling== effectively,** or **using ==async commits== for non-critical writes.**


# Conclusion
- Modern hardware capabilities have fundamentally changed the calculus of system design.
- While distributed systems and horizontal scaling remain necessary for the world's largest applications, many systems can be significantly simpler than what traditional wisdom suggests!

**==UNDERSTAND THE FOLLOWING==**
- Single databases can handle **Terabytes of data**
- Caches can **hold entire datasets in memory**
- Message Queues are **fast enough for synchronous flows** (as long as there's no backlog)
- Application servers have **enough memory** for significant local optimization!












