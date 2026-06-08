---
aliases:
  - Remote Dictionary Server
---
An in-memory datastructure database (think: "Networked data structure server"). A very fast server that stores named keys, where each key points to some typed value such as a hash, list, set, sorted set, stream, JSON document, time series, or vector index.
> "An in-memory store with built-in [[Replication]], persistence, complex data types, and atomic operations"
- It's highly useful in a variety of use cases:
	- [[Cache]]
	- [[Message Queue]]
	- Streaming Engine
	- [[Vector Database]]
	- Real-time coordination layer

Instead of asking Redis to store tables and rows, you ask it to operate on named structures:
```
SET session:abc123 "user-42" EX 3600
HSET user:42 name "Ada" plan "pro"
INCR pageviews:/pricing
ZADD leaderboard 9812 "user-42"
XADD events:orders * order_id 1001 status paid
```
Each command is small, direct, and usually maps to a specific data-structure operation. 

# How Redis Words
- Redis stores its active dataset primarily in ==RAM==.
- Clients connect over the ==Redis protocol==, send commands, and receive replies. The protocol supports pipelining, so clients can send many commands before waiting for responses, reducing network round trips.
- Redis command execution is mostly ==single-threaded==: one main execution path processes commands sequentially, which avoids locking overhead and makes individual commands atomic.
	- Modern Redis can use background threads for some I/O work, but command execution still follows the single-threaded mental model.

- Data is already in-memory
- Commands are simple and data-structure aware
- The server avoids heavyweight query planning
- The event loop is efficient
- Atomic single-command operations avoid many coordination costs (e.g. [[Lock|Locking]])

### Keys and Values
- Every Redis value is stored under a key, which conventionally are colon-separated names:
```
user:42
session:abc123
rate-limit:ip:203.0.113.10
cart:user:42
```
- So Redis doesn't have real namespaces in the SQL schema sense; the colon convention is just naming discipline, but matters a lot for maintainability.
- Keys can ==expire==: `SET <key> <value> EX 900` stores a token for 15 minutes (900 seconds).


### Data Structures
- Strings
- Hashes: field-value maps under one key (user profiles, object metadata, compact records)
	- (In Redis parlance, this doesn't refer to a hash value, it means a map/dictionary/object stored under one Redis key. Use Hashes when the object is flat, and use Redis JSON when the object is deeply nested.)
- Lists: Ordered sequences (simple queues, recent activity, stacks)
- Sets: Unordered unique members (membership checks, tags, deduplication)
- Sorted Sets: Unique members sorted by some numeric score (leaderboards, [[Priority Queue]], rankings)
- Streams: Append-only event logs with consumer groups. Useful for event processing, background work, audit trails, and moderate streaming workloads.
- JSON: Structured document storage and querying
- Bitmaps/bitfields: Compact boolean or integer tracking, such as daily active users
- [[HyperLogLog]]: Approximate cardinality, such as unique visitors.
- [[Bloom Filter]] and [[Cuckoo Filter]]: Probabilistic membership checks, such as "have we probably seen this before?"
- Time series: Timestamped metrics and samples
- Geospatial indexes: For location lookup by radius or distance (e.g. [[Proximity Search]])
- Vector sets/search: For [[Vector Search|Semantic Search]]

# Redis Key Best Practices

Oftentimes a good Redis key can encode:
```
<domain>:<version>:<entity-type>:<entity-id>:<thhing>
```
The version gives us a cheap way to invalidate or change the meaning of a whole group of keys without deleting them one by one.
Examples:
- Caching
	- `cache:v1:user:123:profile`
	- `cache:v1:user:123:settings`
	- `cache:v1:product:sku_ABC123:detail`
	- `cache:v1:org:42:billing-plan`
	- `cache:v1:post:984:comments:page:1
- Sessions
	- `session:sess_abc123`: Stores a logged-in user session, usually has a TTL.
- Permissions
	- `authz:v1:user:123:permissions`: Stores computed permissions/roles for a user.
- External API responses
	- `api:v1:weather:zip:94107`: Caches a response from an external API
- An HTTP response from our application
	- `http:v1:GET:sha256:8f14e45fceea...`: Here, you usually hash the full URLU/query/body instead of putting a giant URL in the key. The value is the response -- either just the body or a full JSON serialization of the status/headers/body.
- A DB query result:
	- `query:v1:user-orders:user:123:page:2:sort:created_desc`: The query + parameters in key, mapped to result.
- Rate-limiting counters:
	- `rl:v1:expensive-service:user:123:60s`
	- `rl:v1:login:ip:203.0.113.10:60s`
- Distributed lock keys, usually set with `NX` and a short TTL:
	- `lock:v1:invoice:555`
- [[Idempotency|Idempotency Key]]s to remember that some event was processed:
	- `idem:v1:stripe-webhook:event:evt_123`
	- `idem:v1:payment:req_7xK9` : A client-provided Idempotency-Key, with a value that looks like the below, so that if the same idempotency key comes in again, you compare the new request hash to the stored requestHash.
		- If the client doesn't provide a key, the server can derive one from stable request fields, hashing them and creating a redis key like `idem:v1:payment:user:123:sha256:9f86d...`, with the value shown below
		- In both cases, the hash should be computed from some **canonical representation**; the same request should produce the same bytes before hashing.
```
# Value in the Client-provides-Idempotency-Key scenario
{
    "status": "completed",
    "requestHash": "sha256:abc123...",
    "response": {
      "paymentId": "pay_456",
      "status": "succeeded"
    },
    "createdAt": "2026-06-08T20:40:00Z"
}

# Value in the Cilent-dosn't-provide-key scneario
{
    "status": "completed",
    "response": {
      "paymentId": "pay_456",
      "status": "succeeded"
    },
    "createdAt": "2026-06-08T20:40:00Z"
}
```
- Cached home feed/timeline
	- `feed:v1:user:123:home`
- Short-lived "user is online" marker
	- `presernce:v1:user:123`

Sometimes for counters or hot-write cases, you might see sharded keys:
- `counter:v1:views:video:999:shard:0`
- `counter:v1:views:video:999:shard:1`
- `counter:v1:views:video:999:shard:2`
And then reads will sum all shards

Note: Keys with `{...}` use the text inside the braces as the hash target!
- `cart:{user:123}:items`
- `cart:{user:123}:total`
These two keys go to the same shard, which is useful if you need multi-key operations.


# Persistence
- Although Redis is memory-first, it *can* persist data to disk for use as a backup. 
- ==There are two classic persistence modes==
	- RDB (Redis Database): Performs compact point-in-time snapshots of your dataset at specified intervals. Good for backups and disaster recovery.
	- AOF (Append-only File): Logs every write operation received by the server. These operations can be replayed again at server startup, reconstructing the original dataset. Commands are logged using the same format as the Redis protocol itself (([[Physical Replication]]?)).
- Redis can use both at the same time! If both are enabled, Redis uses AOF on restart, because that's a more complete reconstruction of the dataset.

# Replication and High-Availability
- Redis supports [[Single-Leader]] replication: A primary accepts writes, and replicas copy the primary's data stream via [[Asynchronous Replication]], meaning acked writes can be lost during failover.
	- Replicas can serve reads in some architectures, support failovers, and provide redundancy.
- ==Redis has two major High-Availability approaches==:
	- Redis Sentinel: Monitors Redis instances and coordinates failover for non-clustered deployments.
	- Redis Cluster: Shards data across multiple primary nodes and can promote replicas during failures.

# Redis Clustering
- Redis Cluster partitions keys across hash ==slots==. Each key belongs to one of 16384 slots, and slots are assigned to nodes. 
- Clients either know the slot map (the mapping of slots to nodes or get redirected to the right node.
- Cluster mode has important implications:
	- Single-key commands work naturally
	- Multi-key operations work only when the relevant keys are in the same hash slot.
		- Hash tags like `user:{42}:profile` and `user:{42}:cart` can force related keys into the same slot.
- Cluster uses asynchronous replication, so failover can lose recent writes in some failure windwos.

# Transactions and Atomicity
- Every single Redis command is atomic. If you run `INCR counter`, no other command interleaves halfway through that increment.
- Redis also has transactions using `MULTI`, `EXEC`, `DISCARD`, and `WATCH`. Transactions queue commands and execute them sequentially as one isolated block (you don't need locks if you're doing single-threaded execution and just bunch your commands together in execution order!).
- Redis also supports [[Lua]] scripting and REdis functions for service-side logic. These are useful when you need multiple reads/writes to happen atomically without round trips.

# Eviction and Memory Management
- Redis is often configured with a `maxmemory` limit. When memory fills, Redis can reject writes or evict keys depending on policy.
- Common eviction policies include:
	- `noeviction`: Return errors when memory is full
	- `allkeys-lru`: Evict [[Least Recently Used]] keys. Common for caches.
	- `allkeys-lfu`: Evict [[Least Frequently Used]] keys. Common for caches.
	- `allkeys-random`: Evict random keys
	- `volatile-lru`, `volatile-lfu`, etc: Evict only keys with TTLs


# Messaging: Pub/Sub vs Streams
- Redis has ==two different messaging styles:==
	- Pub/Sub is fire-and-forget broadcasting. Subscribers receive messages while connected. Good for live notifications, invalidations, chat fanout, and lightweight event propagation.
	- Streams are persistent append-only logs. Consumers can read, ack, retry, and use [[Consumer Group]]s. Streams are better for job processing, event pipelines, and systems that need replay or at-least-once delivery.

# Practical Applications of Redis
- [[Cache]]: Store database query results, rendered pages, API responses, permissions, product catalogs
- [[Session]] Store: Keep login/session data with TTLs across many app servers
- [[Rate Limiting]]: Use `INCR` plus expiration, sorted sets, or Lua scripts for token buckets/sliding windows
- Leaderboards: Sorted sets make ranking natural
- [[Message Queue]]s: Lists for simple queues, streams for durable/observable queues
- Real-Time Analytics: Counters, [[HyperLogLog]], bitmaps, time series
- Deduplication: Sets or [[Bloom Filter]]s
- Feature flags/config: Fast shared reads across services
- [[Distributed Lock]]s: Possible, but must be designed carefully; avoid casual locking for critical correctness
- Presence systems: Track online user with expiring keys or sets
- Shopping carts: Hashes or JSON documents
- Geospatial apps: Nearby drivers, stores, devices, events 
- Search and filtering: Redis Query Engine / ReiSearch-style indexing
- AI/RAG systems: Vector search, semantic cache, conversation memory, retrieval context

Caching Patterns:
- The most common is [[Cache-Aside]], where the application server asks redis for `product:123`, and if it's not found, goes to the backing datastore, gets it, and places the entry in the cache.
- Other patterns:
	- [[Write-Through Cache|Write-Through]]: Write to cache and database together
	- [[Write-Back Cache|Write-Behind]]: Write to cache first, persist later; faster but riskier
	- [[Refresh-Ahead]]: Refresh hot keys before expiration
	- Negative Caching: Cache "not found" briefly to avoid repeated expensive misses
	- Dogpile prevention: Use locks or [[Stale-While-Revalidate]] so that many clients don't regenerate the same value at once in a [[Cache Stampede|Thundering Herd]].


# Footguns
- Running `KEYS *` in production on large datasets; use `SCAN`
- Storing huge values that block the event loop
- Letting lists, streams, or sorted sets grow without bounds
- Using Redis as the only durable store without persistence, backups, and failover design
- Forgetting that async replication can lose recent writes during [[Failover]]
- Putting unrelated workloads with different eviction needs into one Redis database
- Creating hot keys that one shard must handle alone
- Running slow Lua scripts or large O(N) commands on busy instances
- Treating Redis Cluster like a relational database with arbitrary cross-key joins


__________________


SDIAH Deep Dive: https://www.hellointerview.com/learn/system-design/deep-dives/redis

System design is certainly about solving problems with end-to-end, but the interviewers are going to ask you some questions to test you knowledge. We're trying to get you a start to the foundation here...

Why are we learning about Redis to begin with?
- A versatile technology: a lot of bang to your buck!
	- Caches
	- Distributed Logs
	- Leaderboards
	- Replacement for Kafka in certain instances
- Very simple to understand! The conceptual model is quite simple, so you can both explain your choices and understand your implications.
	- It's hard to dive into the details of how a SQL query planner in Postgres works, you're probably screwed; for Redis queries, it's actually pretty straightforward for you to reason through.
	- So this ought to be a good return on your time.

Let's talk about how Redis works from an end-user perspective, then how it operates under the covers, and finally how you can use it in various system design scenarios.

Redis is a **==single-threaded==**, **==in-memory==** **==datastructure server==.**
- Single Threaded: In many databases, the order of operations is hard to grok; in Redis, it's quite simple! The first request to come in is the first one that runs, and everyone else waits! This makes Redis simple to understand.
- In-Memory: It's lightning fast, able to respond in sub-millisecond times, especially for operations like sets and gets. It also means that you can't necessarily guarantee the durability of data. With SQL databases, you need to "batch your requests together, or you run the risk of the N+1 problem. With Redis, you can fire off thousands of requests and the server will happily return its results to you."
- Datastructure server: The core of Redis is this key-value dictionary, but Redis values don't necessarily needs to be strings or numbers or binary blobs: They can also be sorted sets, or hashes, or geospatial indexes, or bloom filters!
	- Redis gives you a way to use familiar data structures in a distributed fashion, so that you can use all the knowledge you've accumulated in building up datastructure and algorithms expertise and bring it into a system design context.

![[Pasted image 20250520101542.png]]
Each of these commands operate on a key.
- These commands are ways of manipulating the various data structures. If you look in the Redis documentation, they're organized by the *type* of the data. It would not make sense to call INCR on a hash, for instance.
- XADD for instance is a way of adding an item to a stream. The important thing is that we have a command XADD (where the X prefix means it operates on a stream), and then there's a key, mystream. 
- The **keys** are how Redis handles the multi-node environments.

While you can run Redis on a single node (in a single thread), it can basically run teh command taht swrite out to disk. You can configure this interval (default is one second, meaning Redis can lose some data). If the process fails, when Redis goes down and comes back up, it ought to read from that file and recover gracefully. 
![[Pasted image 20250520101727.png]]
- For most people, what they're going to do is have a **Replica**. A main or master node, and a secondary that's reading from the append-only file/log maintained by the Main. 
	- This works much like [[Change Data Capture]]; when a command is executed on main, it's eventually executed on the Secondary.
- **This constrains us to the write throughput of our single Main instance**. We could scale our read throughput by scaling replicas... but how do we scale writes?
	- This is where the key space comes into the picture; Redis has a notion of a **Slot,** which is basically the hash of a key (he thinks a CRC) modulo some number (16,384 he thinks), and that's the slot that the key occupies.
	- When the cluster isn't resizing, a single Master/Main node will own that slot... and clients should be aware of all of the nodes in the cluster.
		- So if I have a request from users for Foo, our client will take the hash of Foo, determine the slot it occupies, and then decide which node in the cluster we need to route our request to.
		- Nodes in the cluster communicate through a [[Gossip]] protocol, so they all know about the existience of eachother, as well as which keys/slots they have.
		- If you make a request to the wrong host, it will say: "That key doesn't exist here, it's been moved!" But for performance sake, it's way more beneficial if a client knows which host to go to.
		- This is why when you start up a Client in Redis, you make it aware of all of the hosts that are available.
- The only way to [[Sharding|Shard]] Redis is through choosing your keys... and then, when you're choosing how to Shard, you're functionally choosing how to spread your data across the key space.
	- If you have a Cache... one of the major problems is what's called a [[Hot Spot|Hot Key]] problem, where many of your requests are going to the same key.
	- If one of your keys is located on the first Main node, and the aggregate requests to that node exceed what that node can handle, it doesn't matter that you've broken up key/slot space among many hosts; the uneven distribution of traffic to that singular host is going to kill it!
	- **==PATTERN==**: With Redis, one of the simple patterns is to simply append a random number to the key, such that you write that key to multiple hosts, and you can read it from multiple hosts!
		- This provides a crude way to distribute the load across the cluster. If you're thinking about how you scale Redis, you should be thinking about your key space -- this is essential to how you reason about Redis scaling. 

## Use Case: ==Caching==
- The most common use of Redis is to use it like a [[Cache]]. 
	- You might have a database where you need to make heavy queries; maybe they're analytics queries, maybe it's a search index, whatever.
	- We create a Redis Cache off to the side, and our service needing data:
		- First requests information from the Cache
		- If there isn't an entry in there, the service then goes to the Database, grabs it, and stores the result in the cache.
		- This is appropriate in any case where we can tolerate some stalesness or inconsistency (e.g. a system that can accept [[Eventual Consistency]]; This is many/most systems!)
![[Pasted image 20250520102946.png]]
- We need to make sure that our Cache is spreading out the **LOAD** amongst all of the instances stored inside of it, as much as possible.
	- In Redis, the way we do this is to assign keys. We might append values to our keys such that we are evenly distributing requests among all of our Redis caches. 
	- ((Something that they're not explaining is how this appending really works. I get that we append some data to our key (eg our eventId), then hash it and modulo it, and adding this extra information to hot keys will result in the writes being more evenly distributed... but later when I want to get all of the data in my cache for HotKey 123, how do I know where to look? I don't know what random bits of shit I've been appending on every write for HotKey 123! They aren't explaining that yet.))
		- ((The comments seem to say on Youtube that you have to keep track of which keys are "hot" (and thus sharded using the extra character) via some separate metadata, and during query time, if it's a hot key, aggregation is done over all the shards))... ((Another comment says that you can use the KEYS wildcard to get the key, and the random suffix doesn't matter))...((Another comment says you don't append a completely random number, instead appending a random number in a range of 1-10, and then when you read, you try all those numbers and fanout; this is a standard technique))

- Another thing to consider for all Caches are [[Expiration Policy|Expiration Policies]]:
	- The most common way to use redis is to use the EXPIRE command or to attach arguments to your SET and GET operations... such that after a certain amount of time, that item will no longer be readable.
		- You can say "Expire after 60 seconds", if that's your Cache TTL
	- Another way to configure Redis is in its [[Least Recently Used]] (LRU) setting
		- In this version, you'll continue to be able to append keys into your Cache indefinitely until you run out of memory, at which point Redis will evict the least recently used keys. In many cases this is a drop-in replacement for [[Memcached]].

## Use Case: ==Rate Limiter==
- Another way to use Redis is as a [[Rate Limiting|Rate Limiter]]. 
![[Pasted image 20250520103534.png]]
- Say that we have an expensive service, and we want to guard this expensive service from getting lots of requests; maybe the downstream can only accept 5 requests every 60 seconds.
- If we have multiple instances of our **Service** above, we want to make sure that in aggregate we aren't making more than 5 requests over 60 seconds! How do we do this?
	- We talked about the **atomic increment** command earlier, which increments a key if it exists (if it doesn't exist, sets it to 1); it's basically count++
	- Idea: If this value is over the limit (5 here), we don't want to make the request. If it's under that, that means we have space, and we can proceed with the request. The next thing we want to do is to make sure that this key gets expired. We're going to make sure that we expire this key in 60 seconds.
		- This will let requests proceed through, and after 60 seconds, that key gets automatically "zeroed out" and subsequently the service can begin to make request again.
- This doesn't behave particularly well when Rate Limits are under extreme stress. If I had 60 requests I needed to make, then all of my services are hitting Redis at the same time, and I don't have have any ordering enforced, I might be starving one of the services.
- So ==there's a lot here to talk about in a system design interview, how you set the limits, what's most appropriate with respect to asserting fairness, etc.== This is the most basic implementation of a Rate Limiter; there are a lot of other structures that we could use that include Windows and give clients an idea about when they might be next in line, etc.
	- Keep in mind that there are a lot of logistics that can go into this depending on the needs of your system; sometimes something simple is great.


## Data Structures: ==Streams==
- The most powerful and most complicated primitive that Redis offers is its [[Stream]].
	- (See the Kafka post on the utility of disitrubted append-only logs in distributed system design).
- Imagine ==Redis Streams== as being ordered lists of items
	- They're given an ID, which is usually a timestamp of insertion time.
	- These items can have their own keys and values (think of it like JSON objects)
- Use Case: Building an **Async Job Queue** where we want to be able to insert items onto a queue where they're processed **in order** and **reliably** (if an item is inserted into a queue, it is eventually processed).
- In Redis:
	- We create a [[Stream]] in Redis to store these items. We put items into this stream as they're created.
	- We create a [[Consumer Group]],  which you can think of as a pointer in a stream that defines "where we're at". A consumer group keeps track of where in the stream it has to keep processing.
		- ((Note that a consumer group is a Redis primitive to keep track of where we are in the stream; it exists on Redis. The workers however are separate machines))
	- Worker processes can query the Consumer Group for unallocated items; If a worker asks for a piece of work and the CG is pointing at Item 2 and no workers have picked it up, that item is allocated to the worker.
	- Redis has a concept of **claiming**: At any given moment, only one worker can have a **claim** on an item in the consumer group ((stream?)). If the worker fails for any reason, that item can be reclaimed by the consumer group and reallocated to another worker.
		- So the idea of a Redis stream is that it **gives you a way to distribute work among workers in a way that's fault tolerant (partially, since you have the usual caveats about Redis) and very fast.**

You need to be able to handle [[Failure]]s in Redis
- You might choose to use an option like a Fork of Redis like MemoryDB that gives more reliability.
- We might build [[Redundancy]] in by having replications or additional shards... 
- You also want to make sure that you can keep workers allocated work. 
	- Typically, **Worker** processes, while they're processing an item, are also heartbeating back to the consumer group to tell it that "Hey, I'm still working on this!" This way the consumer group isn't snatching back the work item before the workeer has had a chance to finish it.
	- **==WARN:==** But if a worker loses connection to a consumer group, it might continue to process an item while the consumer group reclaims it and hands it off to another worker! So the behavior here is an [[At Least Once]] processing guarantee, which might be fine for your use case.

## Data Structures: ==Sorted Sets==

Example Use Case: **==Keeping a leaderboard of the Top 5 Most-Liked Tweets that contain the word "Tiger"==**
- The ==SortedSet== commands in Redis all start with Z, and their syntax is quite simple:
![[Pasted image 20250520110118.png]]
- If you look at our Tweet Search answer key, you'll see more about how this will come up. The SortedSet commands all start with Z, and their syntax is quite simple.
	- We give a key (tiger_tweets), the **ranking value** (500 likes), and some string identifier (tweet id, here)
	- In the next tweet, we do the same time for a tweet about zoo tigers
	- Remember that we call these sorted sets? This means for any given ID, it can only have one **ranking value**.
		- So if the Tiger Woods tweet got an additional like, we could run the same command with **501** to update it.
	- We can run the command **ZREMRANGEBYRANK** to remove items that are in a specific rank of ranks. In this case, we're removing all but the top 5. 
		- Every time we add a new tweet, we can remove items that aren't in the top 5, basically maintaining a Top-K 

It's important to note that these run in typical sorted list times!
- When we add things to the list, we're eating a Log(N) complexity... So being able to run ZREMRANGEBYRANK keeps the complexity manageable.

This sounds like a pretty good idea, but remember that we're doing this all in a single key (**tiger_tweets**). If we need to manage top-liked keys by specific keywords, we can probably do this...
**OOPS!** But if we were using this to try to keep track of ALL of the tweets, and try to find the most-liked tweets across our entire setup, then we might have a problem, because they'll all sit on a single node.
If we want it to sit on multiple nodes, we need to keep multiple sorted sets, and combine them at the last minute.
- So we take a hash of the twee id, and keep a sorted set of the items that end up at the same node... when we then want to query this, we have to query all of  sorted sets across our cluster (issuing e.g. 16 queries for 16 shards) and then combining the results. 
- This isn't that big a deal, because Redis is pretty fast, but there are certainly limitations.

## Data Structures: ==Geospatial Indices==
 ![[Pasted image 20250520113647.png]]
- [[Geospatial Index]]es are implemented in Redis in a very useful way!
- The use cases for this vary:
	- If you have a big list of items that have locations, and you want to be able to search them BY location, this is a great way to do it!
- The API looks pretty basic:
	- When we want to add an item to a geospatial index, we:
		- GEOADD {indexKeyName} {long} {lat} {itemIdentifier}
	- When we want to search:
		- GEOSEARCH {indexKeyName} {longAnchor} {latAnchor} BYRADIS {radius, e.g. 5 km} {WITHDIST, optional}

**Implementation:** Under the covers, each of these lat/longs are [[Geohash]]ed to give them a numeric identifier. This numeric identifier is the ranking in the sorted set, and then Redis under the cover is calculating the bounding boxes given your radius, and finding the entries in that range in your sorted set.
- The important thing is that this API is super convenient and works in a wide variety of situations!

**WARN**: There are a number of perils associated with this:
- If your items aren't changing location very often, it may be better to keep a **static list** of longitudes and latitudes *in the service that's making these queries* and just calculating the [[Haversine Distance]] for all of them, to your anchor ponit!
	- If we only have 1,000 stores across the globe, that's not much do just do the arithmetic; it's certainly faster than making a network call to Redis.
- Another Problem: The index currently is on a single Key, which means a single Node. If we need to shard this, then we have to think of a way to do this. There are a few natural ways to do this:
	- Can calculate the GeoHash on my side, take some of the most significant bits and use that as part of the key.
	- Can break this out by Continent, if we don't need to do cross-continent lookups (or if I'm by the border, I'll query two, e.g. North America and South America)


### Use Case: ==Pub Sub==
- [[Publish-Subscribe|Pub Sub]] solves for teh unique instance where your services need to be able to address eachother in some sort of reasonable fashion. 
- ![[Pasted image 20250520114636.png]]
- The canonical example of this would be a Chatroom, where User 1 is connected to Server A, and they need to message User 3, who's connected to Server C. How does Server A know that User 3 is on Server C?
	- This question of "Which shard/server is my data on" might famously be handled by [[Consistent Hashing]], using a hash ring.
		- There's a bunch of incremental problems that happen with these hash rings; notably that it's hard to manage the balance between servers, and scaling them up and down requires a bunch of careful orchestration with a service like [[Apache ZooKeeper]].
- Redis has the idea of [[Publish-Subscribe|Pub Sub]], where servers can connect to Redis and announce a publication they'll be making. On that topic, other servers can subscribe! 
	- So User 1 connects to Server A, and Server A tells Redis "I have User 1. Any messages sent to the topic for User 1 should come back to me!". Server C does the same thing with User 3.
	- When Server A wants to send a message to Server C such that User 3 gets it, it will **publish** to the topic of **user_3**. 
		- ==WARN:== Note that Redis PubSub is [[At Most Once]] delivery! So the idea is that the message *might get to a user* and *might not!* So if you need to guarantee that messages eventually arrive, you'll have to try something else. But Redis PubSub is very fast! It operates on a single box, and can scale quite well.
		- This allows us to... if we wanted to, swap User1 and User3 (they could connect to different hosts), and Redis could be the registry that knew which hosts that each user was connected to. This is something that's a little harder to pull off with a consistent hash ring. 
		- If Server A were to go down, we can migrate User 3 to Server B quite quickly, because Server B wlil register its publication to that topic, and for a while, the messages will go to Server a and Server B, but will eventually get their way to user 3 on Server B. The **WhatsApp** key on the site talks about some of this stuff in depth. 

### Use Case: ==Distributed Lock==
- A common use case of [[Redis]] is SD settings is as a [[Distributed Lock]].
	- Occasionally we have data in our system where we need to maintain consistency during updates (e.g. Design TicketMaster), or when we need to make sure that multiple people aren't performing an action at the same time (e.g. Design Uber).
- In Redis, a simple ==distributed lock with timeout== might use the atomic increment (`INCR`) with a TTL.
	- When we want to try to acquire the lock, we run `INCR`. If the response is 1, we own the lock and proceed. If the response is > 1 (i.e. someone else had the lock), we wait and retry again later. When we're done with the lock, we can `DEL` the key so that other processes can make use of it (rather than sit there retrying and waiting for the TTL to expire).
	- More sophisticated locks in Redis can use the [[Redlock Algorithm]] described [here](https://redis.io/docs/latest/develop/use/patterns/distributed-locks/#the-redlock-algorithm)


## Use Case: ==Rate Limiting== (Oops, covered above too)
- A wide variety of rate limiting algorithms are possible. A common algorithm is a fixed-window rate limiter where we guarantee that the number of requests does not exceed N over some fixed window of time W.
- In Redis:
	- When a request comes in, we `INCR` the keey for our rate limiter and check the response.
		- If the response is greater than N, we wait. If it's less than N, we proceed.
		- We call `EXPIRE` on our key so that after some time period `W`, the value is reset.
			- ((If we look at our previous explanation, it doesn't seem like we reset a timer after every request or anything like that.))



________________

# [Redis Deep Dive w/ a Ex-Meta Senior Manager](https://youtu.be/fmT5nlEkl3U?si=A0W6SXiWDrcnE1aF)


Redis is conceptually simple and highly useful! To know why a Redis query is pretty straightforward!
- Let's talk about how it's useful from a developer perspective.
- Then let's talk about how it operates under the cover.
- And then let's talk about its use in System Design scenarios and common pitfalls.



![[Pasted image 20260608085215.png]]
Redis is a ==single-threaded==, ==in-memory== ==data structure server==
- Single-threaded really simplifies things a lot. In many databases, the order of operation of operations is hard to grok, while in Redis, request are [[First-In First-Out|FIFO]] are serially executed!
- In-memory means that it's lightning fast and can operate in sub-millisecond time for common operations like SETs and GETs, but you can't necessarily guarantee the durability of data. With Redis, you can fire off 1000 requests and the server will happily return its results to you in a way that you couldn't with a SQL database.
- Redis values can be strings, numbers, binary blobs, sorted sets, [[Hash]]es, [[Geospatial Index]]es, [[Bloom Filter]]s

Using Redis:
- commands (e.g. `INCR`) are organized by the type of the data. It wouldn't make sense to call `INCR` on a hash, for instance.
- They can get sophisticated`XADD mystream * name Sara surname OConnor` adds an item to a string!
	- This `XADD` command operates on a Stream datatype, and the key is `mystream`

The keys matter because that's how Redis handles multi-node environments ([[Distributed Cache]])

![[Pasted image 20260608085635.png]]
- You can run Redis on a ==Single Node== in a single thread, and it can write commands that successfully execute out to disk. You can configure the interval, I think the default is ~1s, meaning Redis *can lose data.*
	- But in the ideal scenario, if Redis goes down, it can read from that file and recover somewhat gracefully. In practice this only kind of works
- So in practice, people have a ==Replica== setup, where you have a main with a secondary that reads from that append-only [[Log]] of the master. This works sort of like [[Change Data Capture]]. This still limits us to the write throughput of a single node, but lets us scale read throughput with more read-only replicas.
- Redis has an internal concept called a ==Slot==, which is a hash of a number modulo some number (16384 or something). When the cluster isn't resizing, a single master or main will own that slow, and clients should be aware of all of the nodes in the cluster, so clients will take the hash of `foo`, look up the slot it occupies, and then decide which node in the cluster I route my request to.
	- Each node in the cluster (among mains) communicates with eachother via a [[Gossip]] protocol, so they know about the existence of eachother as well as which slots they have. 
	- If you make a request to the wrong host as the client, it will tell you that the key doesn't exist here and has moved.
	- For performance's sake, it's better if a client knows  exactly which host to go to, which is when when you start up a client, you make it aware of all the hosts that are available.
	- This is where the keyspace becomes important: The only way to shard redis is through choosing your keys, and then when you choose how to shard, you're choosing how to spread your data among your keyspace.
	- ==Important==: If you have a [[Hot Spot|Hot Key]] problem... how does this break Redis? If one of your keys is located on Main A, and the aggregate traffic to A exceeds what it can handle, it doesn't matter that the other Main hosts are out there, the uneven distribution of traffic will kill you!
		- ==Solution==: Append a random number to the key so that way you can write the key to multiple hosts, and you can read it from multiple hosts. This provide a crude way to distribute load across the cluster, but if you think about how you scale redis, you should be thinking about your key space.
			- ((My question below is about this))

-----------
Q: Wait, that solution of appending a random suffix (assumedly multiple times) to a hot key doesn't make sense to me for two reasons:
1. How does the client know which replica to route to, when it wants to read the `taylor-swift` key?
2. If the client updates `taylor-swift-12s1f3`, then what about the keys `taylor-swift-fe1f2f` and `taylor-swift-n923pl`? We have divergent state for the same semantic thing!

A: If `user:123:profile` is extremely hot, adding more primaries doesn't help, because the key still belongs to one slot on one primary. Adding a random number changes the physical keys, though, so you have several Redis keys representing one logical thing.

This can mean two very different strategies:

#### Option 1: Split/Shard the Value   (for Write-Hot keys, not Read-Hot)
- If you have a hot counter `views:video:42`, then you might do something like create multiple keys `views:video:42:0..15`.
- So when you do `INCR views:video:42:7`, with `7` being randomly chosen on the client from `0..15`... this means that *reads* must fan out many requests (`GET views:video:42:0`, `views:video:42:1`, ...) and *sum them!*
	- This only works because counters are mergable. There's no single source of truth inside Redis anymore; the logical value is the aggregate of all shards.
- ==Considerations:==
	- This only "works" because counters are mergable, in this situation.
	- It only helps us with Hot Keys that are hot from a ***write*** perspective. If we have a `0..15` suffix, with 15 keys, our write load on any one of these keys is 1/15 what it would be without this strategy, but because we still have to fan out reads to 15 nods, each of them receives 1/1 reads from the normal case, and you suffer from [[Tail Latency]] problems (to the extent that you'll experience this for in-memory reads).

#### Option 2: Duplicate the Sam Value (for a Read-Hot cache)
- You can store the same logical value multiple times (`product:42:copy:0..4`), and reads pick one copy randomly, which distributes read load. 
- ==Danger==: But now writes have the inverse problem from option one: You must update all copies, and that is not atomically consistent across the cluster. One write can succeed on copy 0 and fail on copy 3. Readers may see different values.
	- This is only safe if Redis is being used a cache and the real source of truth is somewhere else (e.g. Postgres), and you can tolerate stale copies via TTLs, version numbers, invalidation, or rewrite-all-copies best effort.


Q: What about for the read case... just adding more [[Replication|Read Replica]]s?

A: Yes, although adding additional read replicas because of a single hot key in a shard can be somewhat overkill.

### Option 3: Just have more read replicas (Read-Hot cache)
- `hot:key` -> `slot 9182` -> `Primary A` (replicated to Replicas `A1, A2, A3`)
- Writes like `SET hot:key value` still go to `Primary A`
- Reads like `GET hot:key` can be spread across `Primary A, Replica A1, Replica A2, Replica A3`
- This avoids the multiple sources of truth problem, because the primary is still the write authority, and the replicas are copies.
- Caveats:
	- Replicas are [[Eventual Consistency|Eventually Consistent]] because Redis uses [[Asynchronous Replication]].
	- This does *not* help write-hot keys
	- We should only use additional read replicas of the shard that owns the key. ((Yes, you can add "uneven" replica counts to different primaries))
	- Your Redis client needs to support replica reads.

----------

# Uses of Redis.


### Cache
![[Pasted image 20260608095653.png]]
- You have a DB where you need to make heavy queries (analytic queries, etc) and you want to be able to scale this.
- We create a Redis cache off to the side and use it in (e.g.) a [[Cache-Aside]] pattern: On read, our app first checks the Redis cache quickly, and if it's not there, we get it from the database and store it in the cache.
- This is appropriate in any case where you can tolerate some staleness and inconsistency.
- Two concerns to address:
	- [[Hot Spot|Hot Key]] issue: Is our cache spreading out the load amongst all of its instances?
		- In Redis, we do this by assigning keys. We might append values to our keys such that we're eveny distributing requests across our Redis cache.
	- Expiration: A common question is "What's the expiration policy in your cache?"
		- The most common way is to use the `EXPIRE` command, which... after a certain time, an item won't be readable. It's a way of setting a [[Time to Live]] for our cache items.
		- You can also configure it in its [[Least Recently Used]] (LRU) setup. In Redis, you continue to be able to append keys to your Cache indefinitely until you run out of memory, and at that point, Redis starts to evict LRU keys from your setup.

### As a Rate Limiter
![[Pasted image 20260608095701.png]]
- We want to guard this expensive service from getting lots of requests (maybe it can only handle 5 requests every 50 seconds). If we have multiple instances of this service, we want to make sure they aren't, in aggregate, making more than 5 requests every 50 seconds. 
- The Atomic Increment `INCR` command increments a key if it exists (if it doesn't exist, it sets it to 1) and returns the final value. Think of is as `variable++`. 
	- If this value is over the limit, I don't want to make my request.
	- If I get the opportunity to make a request, I want to make sure that this key gets expired. 

```
(Every time you make a request, you atomically run both of the following on Redis first)
INCR expensive_ervice_rate_limit # 5
EXPIRE expensive_service_rate_limit 60 LT
```
- EXPIRE line: "Set the key's TTL to 60 seconds only if 60 seconds is less than the key's current TTL"
Example:
```
Request 1: INCR -> 1, TTL set to 60
Request 2 after 10s: INCR -> 2, TTL is 50, EXPIRE 60 LT does nothing
Request 3 after 20s: INCR -> 3, TTL is 40, EXPIRE 60 LT does nothing

at t=60, Redis expires/deletes the key
Request 4 @ t=61: INCR recreates key at 1, EXPIRE sets TTL to 60
```

This is the most basic Rate Limiter structure you can use. Keep in mind that there's a lot of logistics that can go into this depending on the needs of your system.


### Work Queue
![[Pasted image 20260608100626.png]]
- The value of Redis and its data structure server is that its data structures can be very powerful!
- Redis's ==Stream== primitive is pretty powerful. They can be imagined as ordered list of items. They're given an ID as they're inserted, and each item can have their own Keys and Values. Think of them as JSON objects.

Let's say we want to build an async job queue
- We want items in the queue to be processed in-order and reliably.
- To store these items, we can create a Stream.
- Then we can create a [[Consumer Group]], which you can think of as a pointer into the stream which defines where we're at.
	- A consumer group can keep track of where in the stream it has to keep processing.
- Workers can query the Consumer Group for unallocated items. If the Consumer Group is pointing at Item 2, and no one's picked it up, that item can be allocated to a worker.
- There's a final notion around "Claiming." At any given moment, only one worker can have claim to an item on the Consumer Group. If the worker fails... then that item can be *reclaimed* by the Consumer Group and allocated out to an additional worker.
- The Redis Stream gives us a way to distribute work among many workers in a way that's somewhat fault tolerant, and very fast (so you don't have to have a bunch of latency inserted; though IMO network is the big latency vs disk vs memory access in a queue). 
- You need to be able to handle failures of Redis, and there are options like using a fork of Redis (MemoryDB), and you can also build some redundant in by having replications or additional shards... You'll also want ot figure out how you can keep workers allocated the right work. The typical way this is accomplished is that a worker will continue to [[Heartbeat]] back to the Consumer Group so that the Consumer Group doesn't snatch back the work item before the Worker has a chance to finish. Still, if Worker 2 loses connectivity to the Consumer Group, it might continue to process the item while the Consumer Group reclaims the work and hands it off to another worker. So the Consumer Group can only offer [[At Least Once|At Least Once Delivery]], but not Exactly-Once. This means that you want your work to be [[Idempotency|Idempotent]] if possible.


#### Leaderboard
![[Pasted image 20260608101133.png]]
- Let's say we want to keep a leaderboard of the top-5 most-liked tweets containing the word "Tiger"
- The `Sorted Set` commands all start with `Z`, and their syntax is simple: Give a key (our leaderboard), give a ranking value (500 likes), and some string identifier (the tweet id).
- Remember that these are Sorted Sets. For any ID, it can only have one ranking value
	- If "SomeId1" got another like, we'd run `ZADD tiger_tweets 501 "SomeId1"`
- With `ZREMRANGEBYRANK tiger_tweets 0 -5`, we're removing all but the top 5; it's like a constrained max-heap. 
- Every time I add a new tweet, I can remove the ones that are not in there. I'll only get an incremental example in the list when the tweets of it rises to a number of likes that is greater than my top 5.


With the Sorted Set primitive, we can build a bunch of stuff on top!

### Geospatial Proximity Search
![[Pasted image 20260608101609.png]]
- If you have a big list of items that have locations, and you want to be able to search them by location, this is a great way to do it.
- When I want to add an item to my geospatial index, I add the long/lat and an identifier for my item, and them we add it to our index at this `bikes:rentable` key.
- When we want to search the index, we use `GEOSEARCH`, giving the key, an anchor point, and a radius, and a distance!
	- This returns the nearby stations, together with their distance!
- Under the covers, the long/lats are [[Geohash]]ed to give them a numeric identifier. This identifier is a ranking in the sorted set, and Redis under the covers calculates the bounding box given the radius, and finds the entires in the range in your sorted set.
- The important thing here isn't the internals, it's that the API is super convenient and works in a wide variety of different situations.

This isn't without its perils! There are a number of cases where you might not want to do this.
- If your items aren't changing location very often, it might be better to keep a static list of Longitudes and Latitudes in the service itself, and calculate the haversine distance for all of them. For 1000 locations, that's not too much! This is certainly faster than making a network call out to Redis.
- This index is on a single key, which means it's on a single node.


### [[Publish-Subscribe|Pub Sub]]
- When your servers need to be able to address eachother in some reasonable fashiosn
- ![[Pasted image 20260608102433.png]]
- Chatroom context: 
	- Say User 1 is connected to Server A, and User 3 is connected to Server C... How does A talk to C?
		- Lots of typical solutions: One is to use a [[Consistent Hashing]] ring so that User 3 is always allocated to Server C, and Server A knows that... but there are a bunch of incremental problems that happen with these consistent hash rings (hard to manage load balance, reshuffling is a operational pain)
	- Redis has a PubSub capability!
		- The servers can connect to Redis and announce a publication that they're going to be making.
		- On that Topic, other servers can subscribe.
		- If, for instance, User 1 connects to Server A, Server A tells Reids: "I have user 1, any messages for User 1 come back to me". Server C does the same thing. When Server A wants to send a message to Server C, it publishes to the *topic* of User 3.
		- It's [[At-Most-Once Delivery|At Most Once]] delivery; messages might or might not get to users. This is surprisingly useful in spite of its reliability issues. From a SD perspective, if you need to guarantee that messages arrive, you'll have to try something else.
		- But it's very fast! All it does is bouncing requests between services.








___________


https://youtu.be/k8_qxgoZ4bg?si=vdWUH5kkbm2eD9w_







