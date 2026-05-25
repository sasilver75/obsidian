May 2026, run-through of [SDIAH](https://www.hellointerview.com/learn/system-design/in-a-hurry/introduction)


# [Introduction](https://www.hellointerview.com/learn/system-design/in-a-hurry/introduction)
- System design interviews assess your ability to take an ==ambiguously defined, high-level problem and break it down== into the pieces of infrastructure that you'll need to solve it.
- It's ==not about getting to a single right answer; there are many right answers.==
	- Interviewers assess your ability to navigate complex problems, reason about tradeoffs, and communicate your thinking clearly.
	- Mid-level engineers might cover the basics well and not get into great depth, while seniors work through the basics quickly, leaving time for them to show off the depth of their knowledge in deep dives.
- Each company has different rubrics for system design, but these rubrics have commonalities:
	1. ==Problem Navigation==: Can you navigate a complex, un-specced problem by breaking it down into smaller, more manageable pieces, prioritizing the important ones, and navigating through them to a solution?
		- Typical failure modes:
			- Insufficiently exploring the problem and gathering requirements.
			- Focusing on uninteresting/trivial aspects of the problem, versus the most important ones.
			- Getting stuck on a particular piece of the problem, and not being able to move forward.
			- Failing to deliver a working system
		- These failures are typically due to a lack of structure. We recommend following the structure outlined in the ==Delivery Framework== section, to give yourself a track to run on.
	2. ==Solution Design==: With a problem broken down, your interviewer wants to see you solve each piece of the problem. This is where your knowledge of the ==Core Concepts== comes into play. You should be able to describe how to solve each piece of the problem, and how they fit together into a cohesive whole.
		- Typical failure modes:
			- Not having a strong understanding of the core concepts to solve the problems.
			- Ignoring scaling and performance considerations.
			- Spaghetti design; solutions that are not well-structured or difficult to understand.
		- Interviewers are on alert for candidate who have simply memorized answers or material; they'll test you by probing your reasoning, doubting your answers, and asking you to explore tradeoffs.
	3. ==Technical Excellence==: Knowing about best practices, current technologies, and how to apply them. Knowledge of key technologies and recognized patterns is important.
		- Typical failure modes
			- No knowing about available technologies
			- Using antiquated approaches or being constrained by outdated hardware constraints
			- Not knowing how to apply those technologies to the problem at hand
			- Not recognizing common patterns and best practices
		- Some system design material is still stuck in 2015. Learning the ==numbers to know== will help you make better decisions.
	4. ==Communication and Collaboration==:  These interviews are a great way to get to know what it would be like to work with you as a colleague. Interviews are frequently collaborative, and your interviewer will be looking to see how you *work with them* to solve the problem.
		- Typical failure modes: 
			- Not being able to communicate complex concepts cleanly.
			- Being defense or argumentative when receiving feedback.
			- Getting lost in the weeds and not being able to work with the interviewer to solve the problem.


![[Pasted image 20260520004305.png]]
You need ==practice== to ensure that you're comfortable with these technologies on the day of your interview!


# [How to Prepare](https://www.hellointerview.com/learn/system-design/in-a-hurry/how-to-prepare)

1. Understand what a system design interview IS: Watch videos of mock system design interviews.
2. Choose a delivery framework; System design interviews need to move fast, and it's good to have a clear roadmap.
3. Start with the basics. If you're new to SD, you'll want to start with learning the basics, and mapping out the required knowledge. Core Concepts, Key Technologies, and Common patterns will help you build the mental model that's necessary.

PRACTICE, PRACTICE, PRACTICE: Once you have the foundations, it's time to practice. Passive consumption is good, but you'll retain 10x more if you actually apply it.
1. Choose a question
2. Read the requirements
3. Try to answer on your OWN
4. Read the answer key
5. Put your knowledge to the test, and run a peer mock with others; telling your design out loud under time pressure is a different skill than reading about it.


# [Delivery Framework](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery)
- The easiest way to lose is to not deliver a working system.
- The most often reason why this happens for mid-level candidates is ==time management;== you don't always need to work twice as fast, sometimes you just need to focus on the right things.
- The ==Delivery Framework== is a sequence of steps and timings we recommend for your interview. By structuring your interview in this way, you'll stay focused on the bits that are most important to your interviewer.
	- It keeps you from getting stuck and can ensure you deliver a working system.

![[Pasted image 20260520005321.png]]
Specifically:
- Functional Requirements
- Non-Functional Requirements
- Capacity Estimation
- Core Entities
- API or System Interface
- (Optional) Data Flow
- High-Level Design
- Deep Dives

(FNCC ADHD)

### Functional Requirements (~5 minutes, together with NFR)
- Your =="Users/Clients should be able to..."== statements.
- This is often times a back-and-forth with your interviewer, ask targeted questions as if you were talking to a product manager ("Does the system need to do X?" "What would happen if Y?") to arrive at a prioritized list of features
- Example (Twitter):
	- "Users should be able to post Tweets"
	- "Users should be able to follow other users"
	- "Users should be able to see tweets from users they follow"
- ==Be targeted!== Many of these real systems have hundreds of features; it's your job to identify and prioritize the top three or so. Having a long list of requirements will hurt you more than help you. Focus on what matters!

### Non-Functional Requirements (~5 minutes, together with FR)
- Statements about the system qualities that are important to your users. These can be phrased as =="The system should be able to..."== or =="The system should be..."== statements.
- Example (Twitter):
	- The system should be highly available, prioritizing availability over consistency.
	- The system should be able to scale to support 100M+ DAU (Daily Active Users)
	- The system should be low latency rendering feeds in under 200ms
- It's important that non-function requirements are quantified when possible. "The system should be low-latency" is obvious and not very meaningful; "The system should have low latency search, <500ms" is much more useful, as it ==identifies the part of the system that most needs to be low latency, and provides a specific target.==
- Here's a checklist of items that might be useful in identifying the important non-functional requirements of your system.
	1. ==[[CAP Theorem]]==: Should your system prioritize [[Consistency]] or [[Availability]]?
	2. ==Environment Constraints==: Are there any constraints on the *environment* in which your system will run? Are you running on a mobile device with limited battery life? Running on devices with limited memory or bandwidth (e.g. streaming video on 3G)?
	3. ==Scalability==: Does it have *unique* scaling requirements? For example, does it have bursty traffic at a specific time of day? Are there events like holidays that will cause significant increases in load? Consider the read vs write ratio; does your system need to scale reads or writes more?
	4. ==Latency==: How quickly does the system need to respond to user requests? Specifically consider any requests that require meaningful computation (e.g. low latency search, for Yelp)
	5. ==Durability==: How important is it that the data in your system is not lost? While a social network can likely tolerate some data loss, but a banking system cannot.
	6. ==Security==: How secure does the system need to be? Consider data protection, access control, and compliance with regulations.
	7. ==Fault Tolerance==: How well does the system need to handle failures? Consider redundancy, failover, and recovery mechanisms.
	8. ==Compliance==: Are there legal or regulatory requirements the system needs to meet? Consider industry standards, data protection laws, and other regulations.

### Capacity Estimation
- Many guides will suggest doing back-of-the-envelope calculations at this stage.
	- ==WE BELIEVE THIS IS OFTEN UNNECESSARY.==
- Perform calculations only if they will directly influence your design. In most scenarios, you're dealing with a large, distributed system, and it's reasonable to assume as much.
- Most candidates calculate storage, DAU, and QPS, only to conclude: "Ok, so it's a lot, got it." This doesn't tell interviewers anything.
- When would it be necessary?
	- If you're designing a TopK system for trending topics in FB posts, you'd want to estimate the number of topics you would like to see, as this will influence whether you can use a single instance of data structures like min-heap or if you need to shard it across multiple instances, which will have a big impact on design.

### Core Entities (~2 minutes)
- Next time, you should take a moment to identify and list the ==core entities== of your system. 
- This helps to define terms, understand the that central to our design, and gives a foundation to build on.
- These are the core entities that your API will exchange, and that your system will persist in a Data Model. In the actual interview, ==this is as simple as jotting down a bulleted list and explaining this is your first draft to the interviewer==.
	- Don't fully flesh out the data models at this point; you don't know what you don't know.
	- As you design your system, you'll discover new entities and relationships that you didn't anticipate.
- Once you get to the high-level design, and have a clearer sense of exactly what state needs to update on each request, can start to build out the list of relevant columns/fields for each entity.
- ==Useful questions to ask yourself:==
	- Who are the actors in the system? Do they overlap at all?
	- What are the nouns or resources necessary to satisfy the *functional requirements*?

### API or System Interface (~5 minutes)
- Before you get into high-level design, you'll want to define the contract between your system and its users!
	- ==This often maps directly to the *functional requirements* you've already identified(but not always!)==
- You'll use this contract to guide your high-level design and to ensure that you're meeting the requirements you've identified.
- Which API protocol should you use?
	- [[Representational State Transfer|REST]]: Uses HTTP verbs (GET/POST/PUT/DELETE) to perform CRUD operations on resources. 
	- [[GraphQL]]: Allows clients to specify the data they want to receive, avoiding over/under fetching. You *can* choose this if you have diverse clients with different data needs, but it's gotten somewhat less popular in the last few years.
	- [[Remote Procedure Call|RPC]]: Action-oriented protocols (like [[gRPC]]) that are faster for service-to-service communication; use for internal APIs when performance is critical.
- Don't over think it: ==Default to REST unless you have a specific reason not to.==
	- For real-time features, you also need [[WebSockets|WebSocket]]s or [[Server-Sent Event]]s (SSEs), but design your core API first!

For twitter, might look like:
```
POST /v1/tweets
body: {
  "text": string
}

GET /v1/tweets/{tweetId} -> Tweet

POST /v1/follows
body: {
  "followee_id": string
}

GET /v1/feed -> Tweet[]
```
Above: Notice that we use ==plural resource names== (tweets, not tweet). The current user is derived from the authentication token (e.g. [[JSON Web Token|JWT]]) in the request header, not from the request bodies or path parameters.
- ==Never rely on sensitive information like userIDs from request bodies when they should come from authentication.==

### Data Flow (Optional) (~5 minutes)
- For some backend systems, especially data-processing ones, it can be helpful to describe the high-level sequence of actions or processes that the system performs on the inputs to produce the desired outputs. 
	- If the system doesn't involve a long sequence of actions, skip this!
- Usually defined as ==simple list==, which is used to inform your *high-level design* in the next section.
- For a web crawler, it might look like:
	1. Fetch seed URLs
	2. Parse HTML
	3. Extract URLs
	4. Store data
	5. Repeat

### High-Level Design (~10-15 minutes)
- Now you have a clear understanding of:
	- requirements
	- entities
	- API of your system
- ...you can start to design the high-level architecture, which ==consists of drawing boxes and arrows to represent different components of your system and how they interact.==
- Components are the basic building blocks like servers, databases, caches, etc.
	- The Key Technologies section will give you a good sense of the most common components to know.
- ==Don't layer on complexity too early, resulting in you never arriving at a complete solution. Focus on a relatively simple design that meets core functional requirements, and then layer complexity to satisfy the non-functional requirements in your deep dives section.==
	- It's fine to naturally identify areas where you *can* add complexity (e.g. caches, message queues) in the high-level design. We encourage you to note these areas with a simple verbal callout and written note, then move on.
- As you draw your design, talk about your thought process with your interviewer. 
- Be explicit about how data flows through the system, and what state (DBs, caches, message queues) changes with each request, starting from API requests and ending with the response.
	- When the request reaches your DB/persistence layer, it's a ==great time to start documenting the relevant columns/fields for each entity.==
		- You can do this right next to your database, visually. ==No need to worry too much about types, your interviewer can infer, and they'll only slow you down.==
- Don't waste your time documenting every column/field in your schema. If you have a user table, the interviewer knows that it has a name, email, and password hash. ==Focus on the columns that are particularly relevant to your design.==

Twitter Example:
![[Pasted image 20260520013803.png]]
Above: ==Building up the design, one endpoint (~functional requirement) at a time==

### Deep Dives/Low Level Design
- A simple, high-level design of Twitter is going to be woefully inefficient when it comes to fetching users' feeds. No problem! We handle this in the deep dive section.
- Here, we ==harden our design by making sure it addresses non-functional requirements, addresses edge cases, identifies and addresses issues and bottlenecks, improves design based on probes from interviewer==.
- The degree to which you're proactive here is a function of your seniority.
- Talking about horizontal scaling, introducing caches, database sharding, etc... Things like fanout-on-read vs fanout-on-write and the use of caches.
- Make sure you give your interviewer room to ask questions and probe your design.


# Core Concepts
- The fundamental principles and techniques that form the foundation of every system design interview.
- The ==vocabulary and grammar of system design.==
	- Before understanding how to scale IG or design a ride-sharing service, you have to understanding:
		- What caching is 
		- When to shard a database
		- How networks actually work

#### Networking Essentials
![[Pasted image 20260524135110.png]]
- You can go incredibly deep on networking, but for SD interviews, you need to know the practical bits that come up when designing distributed systems. At a basic level, you need to understand how services talk to eachother, and what happens when those connections fail or get slow.
- The most important decision you'll make is choosing your communication protocol.
	- For most systems, this is [[HTTP]] over [[Transport Control Protocol|TCP]]. Well-understood, works everywhere, and handles 90% of use cases.
	- [[WebSockets|WebSocket]]s and [[Server-Sent Event|SSE]] come up when you need real-time updates.
		- SSE is unidirectional, while WebSockets are bidirectional.
		- [[Server-Sent Event|SSE]]: Client makes an initial HTTP request to open the connection, then server pushes data down that connection (like live scores or notificaitons). The client can't send additional data over the same SSE connection.
		- [[WebSockets|WebSocket]]s are necessary when clients need to push data back into hte server frequently.
		- Both are ==stateful connections== which means ==you can't throw them behind a standard load balancer.== 
			- You'll need to think about connection persistence and what happens when a server goes down with thousands of active connections.
		- Also consider HTTP [[Long Polling]] as an alternative.
	- [[gRPC]] is worth mentioning for internal service-to-service communication, when performance is critical. 
		- Uses a binary serialization and [[HTTP 2]], making it significantly faster than typical [[JSON]] over [[HTTP]].
		- ==Won't use for public-facing APIs== because browsers don't typically natively support gRPC. 
		- A common pattern is REST for external APIs and gRPC internally.
- [[Load Balancing]] is another area that interviewers love to probe.
	- [[Application Layer|Layer 7]] load balancers operate at the application level and ==can route requests based on actual HTTP request content==. You can send API calls to one service and web page requests to another.
	- [[Transport Layer|Layer 4]] load balancers work at the TCP level, and are ==faster but dumber.== They just distribute requests without looking at the content. For WebSockets, you typically need Layer 4 balancing, because you're maintaining a persistent [[Transport Control Protocol|TCP]] connection.

Geography and Latency matter more than most candidates realize:
- A request from NY to London has a minimum latency of around 80ms just from the speed of light through fiber optic cables, before processing anything.
- This is why [[Content Delivery Network|CDN]]s exist: To serve static content from edge servers.

### API Design
- In almost every system design interview, you'll sketch out the APIs that clients use to interact with your system.
- ==Most interviewers don't care about perfect API design;== they want to see that you can create reasonable endpoints and move on to the harder architectural problems.
	- With that said, SLOPPY design can signal inexperience!
- For 90% of interviews, you'll default to [[Representational State Transfer|REST]], which maps resources to URLs and uses HTTP methods to manipulate them.
	- Think `GET /users/{id}` for getting a user, `POST /events/{id}/bookings` for creating a booking.

==Don't spend too much time designing API details.== Sketch out 4-5 key endpoints in a few minute and move on.

There are a ==few concepts worth mentioning==:
- If you're returning large result sets, you'll need ==[[Pagination]]==.
	- ==Cursor-based== pagination is better for real-time data where new items get added frequently.
	- ==Offset-based== pagination is fine for most cases.
- For authentication, use ==[[JSON Web Token|JWT]] tokens== for user sessions, and ==API keys== for service-to-service calls.
- If your system could get hammered by bots or abuse, mention rate limiting, but don't go deep on any of these unless the interviewer specifically asks.

#### Data  Modeling
- One of those things that sounds simple but has massive downstream effects on your system; what date you store, and how you structure it directly affects performance, scalability, and how painful it is to build and maintain the system.
- ![[Pasted image 20260524142124.png]]
- The first big choice is [[Relational Database]] vs [[NoSQL Database]]
	- Relational databases like [[PostgreSQL|Postgres]] work great when you have ==structured data== with clear relationships, and need strong consistency.
		- You'll hear about [[Normalization]] and [[Denormalization]]. 
			- Normalizing means splitting data across tables to avoid duplication.
				- A `users`, `orders`, and `products` table... each `order` references a `userId` and `productId` instead of copying a full user and product data into every order record, keeping your data consistent.
				- ==Start with a normalized relational model, and denormalize specific hot paths if you identify read performance issues! Don't propose denormalization unless you have a clear reason.==
			- Denormalizing duplicates data to avoid joins, making reads faster. The downside is updates; if a user changes their name, you have to update it in the users table plus every order record that copied it. For read-heavy systems where data rarely changes, this tradeoff if often worth it.
	- NoSQL databases like [[Amazon DynamoDB|DynamoDB]] or [[MongoDB]] shine when you need flexible schemas or want to scale horizontally across many servers without complex joins.
		- Forces you to think different:
			- DynamoDB requires you to design your ==partition key== and ==sort key== based on access patterns.. If you're building a a social med app and your most common query is "get all posts for user X", you'd use `user_id` as the partition key, but now queries like "get all posts mentioning hashtag Y" require scanning the entire table.
				- You have to know your queries upfront and design around them!

### Database Indexing
- [[Indexing]] is used to make DB queries fast!
- ==Without indexes, finding a user by Email means scanning every single row in your user table.==
- The most common is a [[B-Tree]]
	- Keeps  data sorted in a tree structure that supports both EXACT lookups and RANGE queries. 
	- Most relational databases create B-Tree indices by default.
- [[Hash Index]]es are faster for exact matches, but can't do range queries, so they're less common.
- You'll see specialized indexes:
	- Full-text Indexes for Search ([[Inverted Index]]) ("Find users with name starting with Sam")
	- [[Geospatial Index]]es for location queries ("find restaurants in 5 miles")
![[Pasted image 20260524143922.png]]
In interviews: ==Think about your actual query patterns, and propose indexes on the fields you're querying frequently!==
- If you're looking up users by email for authentication, index the email column. 
- If you're doing composite queries like "find events in San Francisco on December 25th", you might need a [[Compound Index]] on both city and date.

For specialized needs beyond what your primary DB supports, you may need external systems
- [[ElasticSearch]] is the go-to- for full-text search (though [[pg_search]] is a common Rust-based Postgres extension)
- For geospatial queries [[PostGIS]] is a popular extension to postgres.

External indexes typically sync from your primary database via [[Change Data Capture]], meaning the search index will lag slightly behind the primary database.
- The data you read from the search index is going to be stale by some small amount, but for MOST search use cases, that's almost always acceptable.


### [[Cache]]s
- Comes up in almost every system design interview, usually when you identify that your database is getting hammered with reads. The idea is simple: 
	- ==Store frequently-accessed data in fast memory ([[Redis]]) so that you can skip the DB entirely for most reads.==
- The performance difference is massive:
	- A Cache on Redis takes 1ms, compared to 20-5ms for a typical DB query. It's a 20-50x speedup, which matters for users.
		- It can also reduce load on the database, letting it handle more write traffic and avoiding the need to scale it up prematurely
			- Still, worry here about the [[Cache Stampede]]/thundering herd problem!
![[Pasted image 20260524143931.png]]
- ==The pattern you'll use 90% of the time== is [[Cache-Aside]] with [[Redis]]:
	- On read, *check the cache first!*
		- If the data is there, return it.
		- If not, query the database and store the result in the cache with a [[Time to Live|TTL]], and return it.
- This works well for most read-heavy systems.
- Still, caches introduce complexity! The hard part is ==[[Cache Invalidation Strategy|Cache Invalidation]]==!
	- ==When a user updates their profile in the database, you need to delete or update the cached copy.==
	- Otherwise, the next read returns stale data.
	- There are ==a few strategies here==:
		- You can invalidate the cache entry *immediately* after writes
		- You can use short [[Time to Live|TTL]]s and accept some staleness
- You also need to think about ==cache failures==:
	- If Redis goes down, every request suddenly hits your database. Can it handle that traffic spike? This is called a [[Cache Stampede]] and can take down your whole system!
	- Some approaches include keeping a small in-process cache as a fallback, using circuit breakers to prevent overwhelming the database, or accepting degraded performance until Redis comes back up.

==(! NOTE !)==: A common mistake is to say "Cache everything!" Cache only data that's read frequently and doesn't change often. If you're caching data that changes on every request, you're just adding latency and complexity for no benefit.

[[Content Delivery Network|CDN]] caching is different: IT's for static assets like images, videos, and JS files served from edge locations close to users. In-process caching works for small values like feature flags or config that change rarely, but for your core application data, external caching with Redis is the default.

_______
Q: What does that cache stampede actually look like? People often make handwaving claims about "Oh, it will take down the database," but what mechanically actually happens?

A: 
- The machine hosting the database is typically sized for a certain number of reads/sec (e.g. 500); you might have a 20x load in the case of cache failure.
- What happens next depends on where the bottleneck hits first:
	- If the application's [[Connection Pool]] is saturated, when a new user request needs the DB, it:
		- Asks the pool for a connection
		- There isn't one free
		- So it waits in a queue
		- If waiting exceeds some timeout (either by the application server or the API gateway), the user's request ultimately fails.
		- In this case, the user requests stall and then time out, while the DB itself might still be alive.
		- Still, if the app server is now holding lots of pending requests which consume resources, if enough pile up, the app tier can also become unhealthy.
	- If the application opens up too many DB connections:
		- If each app instance allows 50 DB connections, and autoscaling creates 100 app instances, then you might have 5,000 possible DB connections.
		- Most databases can't handle that many active connection efficiently; for [[PostgreSQL|Postgres]], there's a `max_connections` limit beyond which new connections get rejected with errors like "too many clients."
		- The DB may start rejecting connections, or spend so much time managing connections that useful work slows dramatically.
	- If queries are running but the DB is overloaded
		- Even with a fixed number of connections, the DB can saturate on:
			- CPU
			- Disk I/O
			- Buffer/cache churn
			- Lock contention
			- Memory pressure
			- Network bandwidth
			- Replication lag
			- Transaction log/write pressure.
		- A query that normally takes 10ms might start taking 2s, then 10s, then timing out.
		- Connections are held longer, so the pool frees connections more slowly, which causes more app requests to wait, creating a feedback loop.

So when people say "it takes down the database," they really mean that the database stops being able to serve useful traffic within acceptable latency, causing timeouts and cascading failures across the app; it doesn't always mean that the database process literally exits.

Solutions:
- [[Backpressure]]
- Bounded connection pools
- Short acquire/query timeouts
- [[Circuit Breaker]]s
- Request coalescing
- Stale cache serving
- Rate limits
- Avoiding automatic retries that multiply the traffic
________


### [[Sharding]]
- Sharding comes up when you've outgrown a single databases and need to split your data across multiple independent servers.
	- This happens when you hit storage limits (TBs), write throughput limits (10ks of writes/second) or read throughput that even replicas can't handle.
![[Pasted image 20260524150715.png]]
- The most important decision is your ==shard key==, which determines how data gets distributed and affects everything else in your design.
	- For a user-centric app like Instagram, sharding by `user_id` keeps all of a user's posts, likes, and comments on a single shard.
		- This makes user-scoped queries fast, because they only need to hit one shard.
		- However, this makes "trending posts across all users" expensive because you have to hit every shard and aggregate results. That's the tradeoff.
- Most systems use ==hash-based sharding== where you hash the shard key and use modulo over the number of shards to pick a shard.
- ==Range-based sharding== can work if your access patterns naturally partition, but it's easy to create ==hot spots== if one range gets more traffic.
- ==Directory-based sharding== uses a lookup table to decide where data lives. It's flexible but adds a dependency and latency to every request, so it's rarely worth it in interviews.

==(! NOTE !)==: The biggest mistake with sharding is doing it too early! Doing it introduces a bunch of complications, and a well-tuned single database with [[Replication|Read Replica]]s can handle *way more* than most candidates think. Before proposing sharding, do the capacity math:
- If you're at 10k writes per second and 100GB of data, you DON'T NEED sharding yet!
- ==Bring it up when numbers justify it, not as a default scaling strategy.==

Sharding creations new problems:
- Cross-Shard [[Transaction]]s become nearly impossible, so you need to design your shard boundaries to avoid them. If a user transfer in your banking app requires updating accounts on different shards, you'll need [[Distributed Transaction]]s or [[Saga]]s, which are complex and slow.
- ==Hot spots== happen when one shard gets disproportionate traffic (e.g. Taylor Swift's shard), and resharding is painful.

Only bring up sharding after justifying why a single database won't work, then clearly state your shard key choice and explain the tradeoff.


----
Sam Reality Note:
- In 2026, AWS RDS's biggest instance is `db.m8gd.48xlarge`, which has 768GB ram and 6x1900GB [[Non-Volatile Memory|NVMe]] [[Solid State Disk|SSD]]s (11.4TB of storage)

==So it's pretty likely that you don't actually need to shard, for most applications.==

-----


### [[Consistent Hashing]]
- Solves a specific problem that comes up with distributed caches and sharded databases. 
- When you use simple hash-based distribution of `hash(key) % N`, ==adding a server changes N==
	- This means that *almost every key now maps to a different server*, so you have to ==move most of your data around!== With millions of database records, this is a disaster!
- [[Consistent Hashing]] ==fixes this in the following manner==:
	- Arranging both servers and keys on a virtual ring:
	- Hash each key and place it on the ring, then the key belongs to the next server you encounter going *clockwise*.
	- When you add a new server, only the keys between that new server and the *previous*  (counter-clockwise) server need to move! When you remove a server, only *its* keys need to relocate to the next server on the ring; everything else stays put.
- Huge improvement!
	- With simple modulo hashing, adding one server to a 10-server cluster means moving ~90% of your data.
	- With consistent hashing, you only move about 10%. This makes it practical to add/remove servers dynamically without causing a massive data migration.
![[Pasted image 20260524155635.png]]
- In an interview:
	- You rarely have to explain how consistent hashing works unless specifically asked. It's enough to say: *"==we'll use consistent hashing to distribute data across cache nodes=="* when talking about a distributed cache, or "*==we'll use consistent hashing for the shard key==*" when talking about database sharding; the interviewer just wants to know that you're aware of the technique.


### [[CAP Theorem]]
- Comes up when you're designing distributed systems and need to make tradeoffs about how your data behaves during failures.
- States that you can only have two of three properties at once:
	- Consistency: All nodes see same data
	- Availability: Every request gets a response
	- Partition Tolerance: System works even when network connections fail between nodes
- In practice, you're choosing between consistency and availability in the presence of faults.
> "If you choose ==consistency==, when a network partition happens, some nodes will refuse to serve requests rather than return potentially stale data. Your system might go down, but when it's up, the data is always correct.
   If you choose ==availability==, every node keeps serving requests even during a partition. Users always get a response, but different nodes might temporarily have different data until the partition heals."
   
   ![[Pasted image 20260524160056.png]]
==For most systems, [[Availability]] is the right default.==
- Users can tolerate seeing slightly stale data (e.g. your Instagram feed being 2 seconds old), but can't tolerate the application being down.
- Social media feeds, recommendation systems, and analytics dashboards all work fine with [[Eventual Consistency]].

==[[Strong Consistency]] only matters when the stale data causes actual business problems.==
- Inventory systems need accurate stock counts or you'll oversell products. Banking systems need correct account balance or you'll allow fraud. Booking systems like Ticketmaster need to prevent double-booking the same seat.

==You don't have to pick one model for you entire system, it's common to have different consistency requirements for different parts of the same application.==
- E-Commerce: Product descriptions and reviews can be eventually consistent, but inventory counts and order processing need strong consistency to prevent overselling.

Note: The [[CAP Theorem]] describes behavior during network partitions, which are relatively rare.
- In normal operation, the real tradeoff is between [[Consistency]] and [[Latency]].
- This is captured by the ==[[PACELC]] Theorem==:
>*"During a Partition, choose Availability or Consistency; Else, choose Latency or Consistency."*
- Even when your network is healthy, choosing strong consistency adds latency because nodes need to coordinate before responding.
- In interviews, when you mention replication or distributed data, your interviewer might ask about consistency. ==The safe answer is eventually consistency unless the problem specifically involves money, inventory, or booking limited resources.==


### Numbers to Know
- You don't need to do back-of-the-envelope calculations at the start of an interview, but it's useful to use them when you need to make decisions ("Should I shard the database? Can a single Redis instance handle the cache load?")
- The trick is knowing which numbers to use! Modern hardware is more powerful than many candidates realize.

- Memory Access: Nanoseconds
- SSD Reads: Microseconds
- Network Calls within Datacenter: 1-10ms
- Cross-continent Calls: 10-100s of ms

![[Pasted image 20260524162708.png]]


# Key Technologies ([LINK](https://www.hellointerview.com/learn/system-design/in-a-hurry/key-technologies))

System design involves assembling the most effective building blocks to solve a problem, so it's crucial to have a good understanding of the of the most commonly used building blocks!

## Core Database
- Almost all SD problems will require you to store some data, and you'll typically store it in a database or [[Blob Storage|Object Storage]].
- Most databases are [[Relational Database]]s or [[NoSQL Database]]s. 

==IMPORTANT==: Some candidates try to ham-fist a verbal comparison of Relational vs NoSQL capabilities, but the reality is that these two technologies have become highly overlapping (e.g. NoSQL offering ACID transactions, Relational databases offering JSONB).

### Relational Databases
- [[Relational Database|RDBMS]] are the most common type of database. They're often used for transactional data and are typically the default choice for a product design interview.
- Things to know about relational databases
	- Joins: Can be a major performance bottleneck in your system, so minimize them where possible.
	- Indexes: A way of storing data in a way that makes it faster to query. Commonly [[B-Tree]] and [[Hash Index]]es, but there are also [[Geospatial Index]]es, [[Inverted Index]]es, etc. for specific applications
	- [[Transaction]]s: A way of grouping multiple operations together into a single [[Atomicity|Atomic]] operation that either succeeds or fails together.
- Most common are [[PostgreSQL|Postgres]] and [[MySQL]]


### NoSQL
- A broad category of databases designed to accommodate a wide range of models:
	- [[Key-Value Database]]
	- [[Document Database]]
	- [[Column-Family Database]]
	- [[Graph Database]]
- Don't use a traditional table-based structure and are often schema-less. This allows NoSQL to typically handle large volumes of unstructured, semi-structured, or structured data, and to scale horizontally with ease.
![[Pasted image 20260524163754.png]]
- Strong candidates for situations where:
	- Flexible Data Models
		- ((Note: If it's not Schema-on-Write, it's often Schema-on-Read; you're just moving the problem around. This matters more for some applications than others.))
	- Scalability ([[Horizontal Scaling]])
	- Handling Big Data and Real-Time Web Apps

NOTE: ==The places where NoSQL databases excel are not necessarily places where Relational databases fail, and vice versa!== Don't make broad statements but instead discuss the specific features of the database you're using and how it will help you solve the problem at hand.

Things to know about NoSQL databases:
- Data models: Most common are key-value, document, column-family, and graph databases
- Consistency Models: Offer various consistency models ranging from [[Strong Consistency]] to [[Eventual Consistency]]
- Indexing: Supports indexing to make data faster to query; most common are [[Hash Index]] and [[B-Tree]]
- Scalability: NoSQL databases scale via [[Horizontal Scaling]] via [[Consistent Hashing]]  and/or [[Sharding]] to distribute data across many servers.










   
   
















































