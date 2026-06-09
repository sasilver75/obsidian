---
aliases:
  - Caching
---
Reading from disk is slow. We can introduce in-memory caches, which hold a subset of our data, to speed up data access.

Like databases, caches can and often should use [[Replication]] and [[Sharding]]


We need to consider:
- [[Cache Read Strategy]] (What do we do when the Cache misses? Does the Cache hit the backing datastore )
	- ((I'm still not convinced that this is the umbrella term to use, or if it fits well.))
	- [[Read-Through Cache|Read-Through]]: Application only reads from the cache. On a miss, the cache itself reads the appropriate data from the database, updates itself, and serves the result to the application.
	- [[Cache-Aside]]: Application first reads from the cache. On a miss, the application fetches the value from the database, stores it in the cache, and returns the result.
- [[Cache Write Strategy]] (What do we do when we want to write data?)
	- [[Write-Through Cache]]:  BOTH the cache and the underlying datastore** **simultaneously**! Ensures that your cache is **consistent** with your backing datastore, but can be **slower** for write operations since you have to do something like [[Two-Phase Commit|2PC]]
	- [[Write-Around Cache]]: Data only written to DB; Data is pulled into the cache on a miss. Faster write and DB is source of truth, but Cache misses can be expensive.
	- [[Write-Back Cache]]: An application writes directly to the Cache, and at some point the Cache asynchronously writes to Database. Super fast, and if you aren't reading from the Cache, you might not see the latest writes, plus list of data loss.
- [[Cache Invalidation Strategy]]: Focuses on data *freshness* (ensuring no data of a certain staleness is served)
- [[Cache Eviction Strategy]]: Focuses on capacity management (freeing up space when cache is full)
- [[Cache Warming]]: Proactively populating the cache (e.g. with content likely to be requested) before it's asked for.
- [[Hot Spot|Hot Key]]s: Specific keys (e.g. Taylor Swift) that receive a large amount of write and/or read traffic.
- [[Cache Stampede]]s: If a Cache is redeployed or if a very hot key expires via its TTL, a large number of application services (in the case of Cache Aside) are likely to all ask the database for a value, which can overload/degrade it.
- [[Content Delivery Network]]: A geographically distributed cache often used for static assets (images, videos)
- [[Cache Hit]]: When the Cache has the data you want, and is able to serve it.
- [[Cache Miss]]: When the Cache does not have the data you want.
- [[Refresh-Ahead]]: Similar to the idea of Cache Warming, but typically performed asynchronously after a user gets a cache hit on a cache entry with a TTL that is near expiring. Proactive.
- [[Stale-While-Revalidate]]: Similar to Refresh-Ahead, but allows for serving of stale/expired content up to a specific point. When stale content is served, an asynchronous background job is kicked off to refresh that cache key.

_____________________
SDIAH

A cache is just a temporary storage that keeps recently  used data handy so that you can fetch it quickly next time.

![[Pasted image 20260605194514.png]]

So where should you cache your data?


One option is an [[External Cache]], which is where the cache (e.g. [[Redis]], [[Memcached]]) runs on its own server, totally separate from your application.
![[Pasted image 20260605194617.png]]
- [[Cache-Aside]]: When your application needs data, it first checks the Cache. If it's not there, it's a [[Cache Miss]], so it fetches the data, stores it in the cache, and then returns it to the user.
	- In this case, all of our application servers share the same Cache, so if one of these application servers refresh data for a key, all benefit.



Another option is an [[In-Process Cache]] cache
It's often overlooked, to be honest, but it can be incredibly effective.
![[Pasted image 20260606123726.png]]
Modern servers run on big machines with plenty of memory. 
==This is important, because it's by far the fastest kind of caching==; we don't have to go and have an expensive network hop, the data is already sitting in the same memory space as our application.

Problem:
- Each application has their own in-process memory; we will have inconsistencies between what's cached among our various application servers.

You probably won't have to bring this up unless you have a use case where ultra-low latency matters (caching config data, small lookup tables that every request depends upon).


Another option is a [[Content Delivery Network]] (CDN) that is a geographically distributed network of caches that are close to your users.
![[Pasted image 20260606123920.png]]
Here, we're optimizing for network latency.
- Without a CDN, every request has to travel from the user/client to the origin server (e.g. in virginia).
- In the picture above, that round trip might take 350ms!
	- With a CDN, that same request can hit an [[Edge Server]] some few miles away, maybe 20-40ms round trip.
- When a user requests an image, that request goes to the nearest CDN edge server. If it's a hit, good. If it's a miss, then the CDN itself goes to the origin server and fetches the data, caches it, and returns to the client ([[Read-Through Cache]]). 
- CDNs can cache more than just static media: they can also cache API responses, HTML pages, etc. But it's most common that you're use these for media delivery: Images, media, static files that you want to load quickly around the world.


Lastly, we have [[Client-Side Cache]]ing
![[Pasted image 20260606124131.png]]
This is perhaps the "fastest" way; here, data is stored directly on the user's device (either in the browser, e.g. [[Local Storage]] or in the app, e.g. kept in memory or even written to the device), which totally avoids network hops for information.
- +: This avoids network ops
- -: You don't have control over it! Freshness, invalidation, etc... you don't control the cache!

You'll only usually see this when you have offline functionality or client-heavy workloads. Things like Strava caching your data locally and syncing it once you're online, for instance.



So how do we interact with caches?

[[Cache-Aside]]: The most common pattern by far
![[Pasted image 20260606124521.png]]
The application checks the cache first; if [[Cache Hit]], return it. If [[Cache Miss]], the application fetches from the database, stores in the cache, and returns it.
- +: We only store data that users are actually requesting
- -: Cache misses add latency


[[Write-Through Cache|Write-Through]]
![[Pasted image 20260606124623.png]]
Applications write directly to the cache first, and the cache synchronously writes that data to the database before returning to the user.
- The write isn't complete until both have been updated.
- So you need a cacheing library/framework for your Cache that knows to support this behavior
	- Tools like [[Redis]] or [[Memcached]] DO NOT natively support this. You'll want to use something like Springcache, Hazelcache.....
- Tradeoff:
	- ==-: You have slower writes, and you pollute your cache with data that might not be read again!==
	- -: Suffers from the [[Dual Write Problem]]: What the Cache write succeeds but the Database logic doesn't? In a distributed system, this perfect consistency between Cache/Database is hard to achieve.

In a System design interview, ==Write-through is much less common to use than Cache-aside==


[[Write-Back Cache|Write-Back]]/Write-Behind caching
![[Pasted image 20260606132103.png]]
Similar to write-through, but instead of updating the DBF synchronously, the cache *asynchronously writes to the database in the background*, typically flushing in bashes later on.
- This make writes must faster than a write-through, but introduces new risk of data loss.
- Use only when high write throughput is more important than immediate consistency (and when you can tolerate data loss), e.g. for some metrics pipelines.'

If you're a novice, I wouldn't suggest using this unless you can strongly justify it. Probably avoid it.


[[Read-Through Cache]]
![[Pasted image 20260606132217.png]]
Very similar to [[Cache-Aside]], but the cache handles the database lookup instead of the application.


[[Write-Around Cache]]
- Basically means just don't write to the cache at all, just write to the backing datastore. 
- Have other ways (on read, etc.) to populate the cache.



[[Cache Eviction Strategy]]: Memory is limited; you typically can't fit all of your data in the cache! so you need to figure out a strategy for deciding what to keep and what to remove.
- [[Least Recently Used]] (LRU): Evict items that haven't been used recently. ==Most common and balanced default.==
- [[Least Frequently Used]] (LFU): Evict items that are use least often, even if they were accessed frequently. Good for highly-skewed access patterns.
- [[First-In First-Out]] (FIFO): Evict the oldest item first. Simple, but ==rarely the right choice, since it doesn't take access patterns into account at all.==
- [[Time to Live]] (TTL): Each item expires after a set time (e.g. 5 minutes). ==Great for data that can go stale==, like API responses.

# Common Issues

[[Cache Stampede]], "Thundering Herd"
![[Pasted image 20260606132746.png]]
- A popular cache entry expires (eg. via a [[Time to Live|TTL]] invalidation strategy)
- Even if that window lasts just a second, each of those cache misses is goign to hit our database, which can overwhelm our database

Example: website where we cash the homepage feed with a TTL of 60 seconds
- We get 100,000 requests every second
- Usually 100,000 goes to the cache
- But after 60s, it expires
- If it expires, in the case of [[Cache-Aside]], our application server goes to the database to ask for the cache
- 100,000 requests come in, cache mis, then hit the database together, overwhelm it.

We can prevent in two ways:
- [[Request Coalescing]]: When a request tries to rebuild the same cache key... only the first one should work, and the rest should just wait for the results to come in, and then read from the cache.
- [[Cache Warming]]: Instead of waiting for popular keys to expire, we can proactively refresh them just before they do. So at the 55 second mark, we can come in, refresh the feed, and prevent it from ever actually expiring.


[[Cache Consistency]], one of the most common issues interviewers liek to ask about
![[Pasted image 20260606135714.png]]
- This happens because the cache and the DBF return different values for the same data
- Most systems read from the cache and write to the database. This creates a window based on your invalidation/eviction policy where you can have stale data!
- Example
	- A user writes a new profile picture for their profile
	- The new value is written to the DB
	- The old one is still sitting out there in the cache
	- Other users requesting our user's profile are still getting the old profile picture
	- They continue reading this old value until the cache record is evicted somehow

	There's no "Fix" to this, and how much you care depends on how fresh your data needs to be.
But there are some strategies:

1. ==Invalidate on Write==: If consistency is really important, then when the profile picture came in, we can go and proactively delete that key from the cache. Later, we'll have a cache miss and pull it in.
2. Use a short [[Time to Live|TTL]]: This minimizes the time where you have cache incoherency.
3. Accept that moderate eventual consistency is fine. If you have a 5 minute TTL, and users see a stale profile picture for 5 minutes, maybe that's totally fine, and life will still go on. This depends on your use csae.


[[Hot Spot|Hot Key]]s: Cache entries that get way more traffic than anything else.
![[Pasted image 20260606140001.png]]
This can become a large bottleneck
- Imagine you're building Twitter, and everyone is viewing Taylor Swift's profile. 
- That cache key could be receiving millions of requests per second. This can overload a single Redis shard, even though the cache is working as expected. 
- Cacheing increases your overall read throughput (using memory instead of disc), but it doesn't totally solve hte problem if you have something so overwhelmingly popular that the read load is huge.

Solutions:
1. ==Replicate Hot Keys!== You can put Taylor Swift on *each* of the shards/caches, so the application server can load balance evenly amongst all of them.
2. ==Add a local fallback cache==. Use an in-process cache in the application server where we keep extremely  hot values... so that repeated requests don't go hit Redis at all. ((Think of downsides to this))


![[Pasted image 20260606140314.png]]
Don't just add a cache for the sake of adding a cache. 
- Candidates often throw it down without really needing it, or without explanation.
- We want to do this when we have a really read-heavy workload. 
	- Many queries
	- Expensive queries (e.g. Newsfeed requires doing a bunch of expensive joins; maybe we cache this newsfeed with a TTL of 60s or so and serve it quickly from something like Redis)
	- If your database is starting to peg out on CPU, it might make sense to add a 
	- Latency requirements from your [[Non-Functional Requirement]]s say that you need (e.g.) 100ms response time for some API endpoint. You can argue or justify that this database query might take too long with expensive queries, ands o you might have to cache it instead.


How to introduce caching:
- Identify the bottleneck
- Decide what to cache: Focus on data that is fetched frequently, doesn't change often, and is expensive to compute.
- Think about what the cache keys should be! What are your cache keys?
	- Common mistake for junior/mid level
- Choose cache architecture: e.g. cache-aside on read, write-around, whatever.
- Set an eviction policy, e.g. an appropriate TTL
- Address the downsides: 
	- Do I have a TTL on a hot key that could result in a cache stampede? How will we address this?
	- Do I have problems with hot keys that can't even be served by a cache? How will we address this?
	- Do I have to think more clearly about cache consistency? Does this matter? How to address this?

