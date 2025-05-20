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
- In-Memory: It's lightning fast, able to respond in sub-millisecond times, especially for operations like sets and gets. It also means that you can't necessarily guarantee the durability of data. With SQL databases, you need to "batch your requests together, or you run the risk of the N+1 problem. With Redis, you can fire off thousands of requests and the server will happliy return its results to you."
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
	- If you have a Cache... one of the major problems is what's called a [[Hot Key Problem]], where many of your requests are going to the same key.
	- If one of your keys is located on the first Main node, and the aggregate requests to that node exceed what that node can handle, it doesn't matter that you've broken up key/slot space among many hosts; the uneven distribution of traffic to that singular host is going to kill it!
	- **==PATTERN==**: With Redis, one of the simple patterns is to simply append a random number to the key, such that you write that key to multiple hosts, and you can read it from multiple hosts!
		- This provides a crude way to distribute the load across the cluster. If you're thinking about how you scale Redis, you should be thinking about your key space -- this is essential to how you reason about Redis scaling. 

Use Cases:
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

- Another thing to consider for all Caches are [[Expiration Policy|Expiration Policies]]:
	- The most common way to use redis is to use the EXPIRE command or to attach arguments to your SET and GET operations... such that after a certain amount of time, that item will no longer be readable.
		- You can say "Expire after 60 seconds", if that's your Cache TTL
	- Another way to configure Redis is in its [[Least Recently Used]] (LRU) setting
		- In this version, you'll continue to be able to append keys into your Cache indefinitely until you run out of memory, at which point Redis will evict the least recently used keys. In many cases this is a drop-in replacement for [[Memcached]].

- Another way to use Redis is as a [[Rate Limiting|Rate Limiter]]. 
![[Pasted image 20250520103534.png]]
- Say that we have an expensive service, and we want to guard this expensive service from getting lots of requests; maybe the downstream can only accept 5 requests every 60 seconds.
- If we have multiple instances of our **Service** above, we want to make sure that in aggregate we aren't making more than 5 requests over 60 seconds! How do we do this?
	- We talked about the **atomic increment** command earlier, which increments a key if it exists (if it doesn't exist, sets it to 1); it's basically count++
	- Idea: If this value is over the limit (5 here), we don't want to make the request. If it's under that, that means we have space, and we can proceed with the request. The next thing we want to do is to make sure that this key gets expired. We're going to make sure that we expire this key in 60 seconds.
		- This will let requests proceed through, and after 60 seconds, that key gets automatically "zeroed out" and subsequently the service can begin to make request again.
- This doesn't behave particulary well when Rate Limits are under extreme stress. If I had 60 requests I needed to make, then all of my services are hitting Redis at the same time, and I don't have have any ordering enforced, I might be starving one of the services.
- So ==there's a lot here to talk about in a system design interview, how you set the limits, what's most appropriate with respect to asserting fairness, etc.== This is the most basic implementation of a Rate Limiter; there are a lot of other structures that we could use that include Windows and give clients an idea about when they might be next in line, etc.
	- Keep in mind that there are a lot of logistics that can go into this depending on the needs of your system; sometimes something simple is great.


## Data Structures: Streams
- The most powerful and most complicated primitive that Redis offers is its [[Stream]].
	- (See the Kafka post on the utility of disitrubted append-only logs in distributed system design).
- Imagine ==Redis Streams== as being ordered lists of items
	- They're given an ID, which is usually a timestamp of insertion time.
	- These items can have their own keys and values (think of it like JSON objects)
- Use Case: Building an **Async Job Queue** where we want to be able to insert items onto a queue where they're processed **in order** and **reliably** (if an item is inserted into a queue, it is eventually processed).
- In Redis:
	- We create a [[Stream]] in Redis to store these items. We put items into this stream as they're created.
	- We create a [[Consumer Group]],  which you can think of as a pointer in a stream that defines "where we're at". A consumer group keeps track of where in the stream it has to keep processing.
	- Worker processes can query the Consumer Group for unallocated items; If a worker asks for a piece of work and the CG is pointing at Item 2 and no workers have picked it up, that item is allocated to the worker.
	- Redis has a concept of **claiming**: At any given moment, only one worker can have a **claim** on an item in the consumer group ((stream?)). If the worker fails for any reason, that item can be reclaimed by the consumer group and reallocated to another worker.
		- So the idea of a Redis stream is that it **gives you a way to distribute work among workers in a way that's fault tolerant (partially, since you have the usual caveats about Redis) and very fast.**

You need to be able to handle [[Failure]]s in Redis
- You might choose to use an option like a Fork of Redis like MemoryDB that gives more reliability.
- We might build [[Redundancy]] in by having replications or additional shards... 
- You also want to make sure that you can keep workers allocated work. 
	- Typically, **Worker** processes, while they're processing an item, are also heartbeating back to the consumer group to tell it that "Hey, I'm still working on this!" This way the consumer group isn't snatching back the work item before the workeer has had a chance to finish it.
	- **==WARN:==** But if a worker loses connection to a consumer group, it might continue to process an item while the consumer group reclaims it and hands it off to another worker! So the behavior here is an [[At Least Once]] processing guarantee, which might be fine for your use case.

## Data Structures: Sorted Sets

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

## Data Structures: Geospatial Indices
 ![[Pasted image 20250520113647.png]]
- [[Geospatial Index]]es are implemented in Redis in a very useful way!
- The use cases for this vary:
	- If you have a big list of items that have locations, and you want to be able to search them BY location, this is a great way to do it!
- The API looks pretty basic:
	- When we want to add an item to a geospatial index, we:
		- GEOADD {indexKeyName} {long} {lat} {itemIdentifier}
	- When we want to search:
		- GEOSEARCH {indexKeyName} {longAnchor} {latAnchor} BYRADIS {radius, e.g. 5 km} {WITHDIST, optional}

**Implementation:** Under the covers, each of these lat/longs are [[GeoHash]]ed to give them a numeric identifier. This numeric identifier is the ranking in the sorted set, and then Redis under the cover is calculating the bounding boxes given your radius, and finding the entries in that range in your sorted set.
- The important thing is that this API is super convenient and works in a wide variety of situations!

**WARN**: There are a number of perils associated with this:
- If your items aren't changing location very often, it may be better to keep a **static list** of longitudes and latitudes *in the service that's making these queries* and just calculating the [[Haversine Distance]] for all of them, to your anchor ponit!
	- If we only have 1,000 stores across the globe, that's not much do just do the arithmetic; it's certainly faster than making a network call to Redis.
- Another Problem: The index currently is on a single Key, which means a single Node. If we need to shard this, then we have to think of a way to do this. There are a few natural ways to do this:
	- Can calculate the GeoHash on my side, take some of the most significant bits and use that as part of the key.
	- Can break this out by Continent, if we don't need to do cross-continent lookups (or if I'm by the border, I'll query two, e.g. North America and South America)


### Capability: Pub Sub
- [[Publish-Subscribe|Pub Sub]] solves for teh unique instance where your services need to be able to address eachother in some sort of reasonable fashion. 
- ![[Pasted image 20250520114636.png]]
- The canonical example of this would be a Chatroom, where User 1 is connected to Server A, and they need to message User 3, who's connected to Server C. How does Server A know that User 3 is on Server C?
	- This is an instance of "Which "
- 
 