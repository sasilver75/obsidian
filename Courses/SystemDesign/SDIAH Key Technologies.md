https://www.hellointerview.com/learn/system-design/in-a-hurry/key-technologies

----------------
![[Pasted image 20250518133847.png|600]]
Most interviewers aren't going to care whether you know about a particular queueing solution so long ans you have one that you can use. But without knowing about **any** queueing solutions, you'll have a hard time designing a system that requires one!
In this section we'll walk through some of the key categories of technologies relevant to solving 90% of system design problems, together with some discussion of the possible options for each. You're generally free to choose which technology you want to use in each category, but we recommend you have at least one.


# Core Database
- Almost all system design problems will require you to store some data, and you're most likely going to be storing it in either a database or in blob storage.
- The most common types are [[Relational Database]]s (e.g. [[PostgresDB|Postgres]]) and [[NoSQL Database]]s (e.g. [[DynamoDB]]). 
	- We recommend picking ONE for your interview.
		- If you're talking predominantly product design interviews, we recommend a relational database.
		- If you're taking predominantly infrastructure design interviews, we recommend a NoSQL database.
	- **==WARNING==**: Many candidates trip themselves up by trying to insert a comparison of relational and NoSQL databases into their answer. The reality is that ==these two technologies are highly overlapping==, and that broad statements like "*I need to use a relational database because I have relationships in my data*", or "*I've gotta use NoSQL because I need scale and performance*" are often ==yellow flags== that reveal inexperience.
		- Interviewers don't expect you to give an explicit comparison of SQL and NoSQL databases, and it's a pothole that you should completely avoid. 
		- Instead, ==talk about what you know about the database you're using, and how it will help you solve the problem at hand.==
			- "I'm using Postgres here because its ACID properties will allow me to maintain data integrity" is a great leader.
#### Relational Databases
- [[Relational Database]]s (RDBMS) are the most common type of database, often used for **transactional data** (e.g. user records, order records) and are typically the default choice for a product design interview.
- Relational databases store your data in tables, which are composed to rows and columns. Each row represents a single record, and each column represents a single field on that record. A user's table might have a name and email column. Often queried via [[SQL]], a declarative language for querying data.
- Relational databases come equipped with several features which are useful for system design interviews:
	- **SQL Joins**:
		- A way of combining data from multiple tables. Joins ==can be a major performance bottleneck== in the system, so ==minimize them when possible==.
	- **[[Index]]es**: 
		- A way of storing data that ==makes it faster to query==. You might create an index on the `name` column of your `users` table, allowing you to query for users by name much quicker than if we didn't have the index.
		- Often implemented using a [[B-Tree]] or [[Hash Map]].
		- Relational databases support arbitrarily many indexes, which allow you to optimize different queries, and provide support for ==multi-column== and ==specialized indexes== (e.g. geospatial indexes, full-text indexes).
	- **RDBMS [[Transactions]]**:
		- ==A way of grouping multiple operations together into a single atomic operation.==
		- If you have a users table and a posts table, you might want to create a new user and a new post for that user at the same time. With a transaction, either both operations will succeed or both will fail, ensuring you don't have invalid data like a post from a user who doesn't exist.

##### NoSQL Databases
- A broad category of databases designed to accommodate a wide range of data models, including [[Key-Value Database]]s, [[Document Database]]s, [[Column-Family Database]]s, and [[Graph Database]]s.
- NoSQL databases ==do not use a traditional table-based structure ,and are often schema-less==, allowing NoSQL databases to handle large volumes of unstructured, semi-structured, or structured data, and to scale horizontally with ease.
![[Pasted image 20250518135453.png|500]]
- NoSQL databases are strong candidates for situations where you need:
	- **Flexible Data Models**: Your data model is evolving or you need to store different types of data structures without a fixed schema.
	- **Scalability**: Application eds to scale horizontally to accommodate large amounts of data or high user loads.
	- **Handling Big Data and Real-Time Web Apps**: You have applications dealing with large amounts of data, especially unstructured data, or applications requiring real-time data processing and analytics.

**==WARNING==**: The places where NoSQL databases excel are not necessarily places where relational databases fail (and vice versa). 
- While NoSQL databases are great for handling unstructured data, relational databases can also have JSON columns with flexible schemas.
- While NoSQL databases are great for scaling horizontally, relational databases can also scale horizontally with the right architecture.
- When you're discussing NoSQL databases in your SD interview, make sure you're not making broad statements but instead discussing the specific features of the database you're using, and how they will help you solve the problem at hand.

Things to know about NoSQL databases:
1. ==Data Models==: NoSQL databases come in variety of flavors, each with its own data model. The most common types are KV stores, document stores, column-family stores, and graph databases.
2. ==Consistency Models==: Offer consistency models ranging from [[Strong Consistency]] to [[Eventual Consistency]]. Strong ensures that all nodes in the system have the same data at the same time, and eventual ensures that all nodes will *eventually* have the same data (with no upper bound on what eventually means).
3. ==Indexing==: NoSQL databases support indexes to make data faster to query; the most common types are B-Tree and Hash Table indexes.
4. ==Scalability==: NoSQL databases scale horizontally by using [[Consistent Hashing]] and/or [[Sharding]] to distribute data across many servers.

The most common NoSQL databases are:
- [[DynamoDB]]: Breadth of features and widely accepted.
- [[Cassandra]]: ==Good choice for write-heavy workloads due to its append-only storage model==, but comes with some tradeoffs in functionality.
- [[MongoDB]]: Document-oriented.

# Blob Storage
- [[Blob Storage]] (often used interchangeably with "Object Storage"), is for when you ==need to store large, unstructured blobs of data==, which could be images, videos, or other files.
- Storing these large blobs in a traditional database is both expensive and inefficient, and should be avoided whenever possible -- instead, use a blob storage service like [[Amazon S3]] or [[Google Cloud Storage]]; these platforms are specifically designed for handling large blobs of data, and are much more cost-effective than a traditional database.
- ==Blob storage services are simple: You can upload a blob of data and you get back a URL. You can later then use this URL to download the blob of data.==
	- Oftentimes blob storage services work in conjunction with [[Content Distribution Network|CDN]]s so that you can get fast downloads from anywhere in the world. Upload a file/blob to blob storage which will act as your origin, then use a CDN to cache the file/blob in edge locations around the world.
- ==**WARNING:**== Avoid using blob storage like S3 as your primary database unless you have a very good reason. ==In a typical setup, you'll have a core database like Postgres or DynamoDB that has **pointers** (just a url) to the blobs stored in S3==. This allows you to use the database to query and index the data with very low latency, while still getting the benefits of cheap blob storage.

Here are some common usages of blob storage:
- Design **Youtube**: Store **videos in blob storage**, store **metadata in a database**.
- Design **Instagram**: Store **images and videos in blob storage**, store **metadata in a database.**
- Design **Dropbox**: Store **files in blob storage**, store **metadata in a database**.

![[Pasted image 20250518141218.png|600]]

==File Uploading Flow==:
- When a client wants to upload a file, they request a **==presigned URL==** from the server.
- The server returns this presigned URL to the client, recording it in the database.
- The client uploads the file to the presigned URL.
- The **blob storage** triggers a **notification** to the **server** that the upload is complete, and the status is updated.

==File Downloading Flow:==
- Client requests a specific file from the server (perhaps by referring to the id of the photo, for instance), and is returned a **presigned URL**.
- The client uses the presigned URL to download the file via the **CDN**, which proxies the request to the underlying blob storage.
	- ((Wait, what? I thought the CDN would have the image stored locally? Maybe the CDN is functioning as a Cache, and on cache miss it pulls the actual binary file from the origin and caches it before returning to the user?))

Things to know about blob storage:
- **Durability**: Blob storage services are designed to be ==incredibly durable==, using [[Replication]] and [[Erasure Coding]] to ensure your data is safe even if a disk or server fails.
- **Scalability**: Services like S3 can be considered ==infinitely scalable==. They can store an unlimited amount of data and can handle an unlimited number of requests (obviously within the limits of your account). Consider the scalability a given, during your interview.
- **Cost**: Cost-effective; ==much cheaper than storing large blobs of data in traditional databases==.
	- AWS S3 charges $0.023/GB/month for the first 50TB of storage. 
	- Dynamo charges $1.25/GB/month for the first 10TB of storage.
- **Security**: Blob storage services have built-in security features like ==encryption at rest and in transit==. They also have ==access control features== that allow you to control who can access your data.
- **Upload and Download Directly from the Client**: Blob storage services ==allow you to upload and download blobs directly from the client!== This is useful for applications that need to store/retrieve large blobs of data, like images or videos. 
	- Familiarize yourself with [[Presigned URL]]s and how they can be used to grant temporary access to a blob, either for upload or download!
- **Chunking**: When uploading large files, it's common to use chunking to upload the file in smaller pieces, which allows us to resume an upload if it fails partway through. This is especially useful for large files, where uploading the entire file at once might take a long time. Modern blob services ==support chunking out of the box, via the **multipart upload API**==.

# Search-Optimized Database
- [[Full-Text Search Index|Full-Text Search]] is the ability to search through a large amount of text data and find relevant results.
- This is different from a traditional database query, which is usually based on **exact match** or **ranges**.
- Without a search-optimized database, you might need to run something like:
```SQL
SELECT * FROM documents WHERE document_text LIKE '%search_term%'
```
- This query is slow and inefficient and doesn't scale well, ==requiring a full table scan, yikes!==
- Instead, Search-optimized databases are designed to handle full-text search, using techniques like:
	- Indexing
	- Tokenization
	- Stemming
- They build [[Inverted Index]]es, which are data structures mapping words to the documents that contain them, allowing you to quickly find documents that contain a given word.
- Instead of scanning the entire table, the database can quickly look up the word in the query and find related documents, fast!

Examples of usage:
- **TicketMaster**: Searching through a large number of events to find relevant results.
- **Twitter**: Needs to search through a large number of tweets to find relevant results.

Things to know about search-optimized databases:
1. **==Inverted Indexes==**: Search-optimized databases use inverted indexes to make queries fast and efficient.
2. ==**Tokenization**==: Breaking a piece of text into individual "words", allowing you to map from words to documents in the inverted index.
3. **==Stemming==**: The process of reducing words to their root form, allowing you to match different forms of the same word. For example, "running" and "runs" would both be reduced to "run."
4. **==Fuzzy Search==**: The ability to find results that are similar to a given search term. Most search-optimized databases support fuzzy-searching out of the box as a configuration option. Achieved through techniques like ==edit distance calculation==, which returns the number of letters that need to be changed/added/removed to transform one word into another.
5. ==**Scaling**==: Just like traditional database, search-optimized databases scale by adding more nodes to a cluster and sharding data across these nodes.

Examples of search-optimized databases:
- The clear leader is [[ElasticSearch]]
	- A distributed, RESTful search or analytics engine built on top of Apache [[Lucene]], designed to be fast/scalable/easy to use.
- Other options include [[PostgresDB|Postgres]]'s use of [[GIN Index]]es ,which support full-text search. [[Redis]] has a quite immature and bad full-search capability.



# API Gateway
- API gateways sit in front of your system and are ==responsible for routing incoming requests to the appropriate backend service==.
	- If a client requests GET /user/123, the API gateway would route that request to the users service and return the response to the client.
- Also responsible for ==handling cross-cutting concerns== like [[Authentication]], [[Rate Limiting]], and [[Logging]].
- **TIP:**==In nearly all product design style system design interviews, it's a good idea to include an API gateway in your design as the first point of contact for your clients.==
![[Pasted image 20250518153646.png|300]]
- Note that interviewers rarely get into detail of the API gateway, instead wanting to ask questions which are more specific to the problem at hand.
- The most common API gateways are **AWS API Gateway**, **Kong**, and (google's) **Apigee**. It's not uncommon to have an **nginx** or Apache webserver as your API gateway.

# Load Balancer
- Most system design problems will require you to design a system that can handle a large amount of traffic.
- When you have a large amount of traffic, you often need to distribute that traffic across multiple machines ([[Horizontal Scaling]]) to avoid overloading any specific machine.
- For the purposes of an interview, you can assume that a Load Balancer is a black box that will distribute work across your system.
	- ==In an interview, it can often be redundant to draw a load balancer in front of every service.== Instead, either omit it from the design (mentioning them), or add one only to the front of the design as an abstraction.
- Sometimes you'll need to have ==specific features== from your load balancer, like [[Sticky Session]] or persistent connections. The most common decision to make is whether to use a [[Layer 4]] or [[Layer 7]] load balancer.
	- ==**TIP:**== If you have persistent connections like [[Websockets]], you'll likely want to use an L4 load balancer. Otherwise an L7 load balancer offers great flexibility in routing traffic to different services while minimizing the connection load downstream.
- Common load balancers:
	- AWS Elastic Load Balancer (ELB), [[NGINX]] (an open-source webserver often used as a load balancer), and [[HAProxy]] (a popular open-source load balancer). For problems with extremely high traffic, specialized hardware load balancers will outperform software load balancers that you'd host yourself, and you'll quickly be pulled into the crazy world of network engineering.


# Queue
- Queues serve as **==buffers for bursty traffic==** or as ==**means of distributing work across a system**==.
	- A compute resource sends **messages** to a **queue** and then forgets about them.
	- On the other end, a pool of **workers** (also compute resources) processes the messages as their own pace.
	- ==Messages can be anything from a simple string to a complex object.==
- The ==queue's function== is to smooth out the load on the system!
	- If I get a spike of 1,000 requests but can only handle 200 requests per second, 800 requests will wait in the queue before being processed, but they aren't dropped!
	- Queues ==decouple the producer and consumer of a system, allowing you to scale them independently!==

**==WARNING==**: Be careful of introducing queues into synchronous workloads! **If you have strong LATENCY requirements (e.g. < 500ms), by adding a queue you're nearly guaranteeing that you'll break that latency constraint**.

Let's look at a couple common use cases for queues:
1. **==Buffer for Bursty Traffic==**: In an app like Uber, ride requests can spike massively during special events. A queue can hold these incoming requests ,allowing the system to process them at a manageable rate without overloading the server or degrading the user experience.
2. **==Distribute Work Across a System==**: In a photo processing service, queues can be used to distribute expensive image processing tasks: A user's might upload a photo for editing, and an according task will be placed in a queue. Different worker nodes them pull tasks from the queue, ensuring even distribution of workload and efficient use of computing resources.

Things you should know about queues:
1. ==**Message Ordering**:== Most queues are ==FIFO==, meaning messages are processed in the order they're received, though ==some queues like [[Kafka]] allow for more complex ordering guarantees, such as ordering based on specified priority or time==.
2. **==Retry Mechanisms==**: Many queues have built-in retry mechanisms that attempt to redeliver a message a certain number of times before considering it a failure. You can configure delay between attempts and max number of attempts.
3. ==**[[Dead Letter Queue]]s**==: These queues are ==used to store messages that cannot be processed==. They're useful for debugging and auditing, as it allows you to inspect messages that failed to be processed, and understand why they failed.
4. ==**Scaling with [[Partition]]s**==: Queues can be partitioned across multiple servers so that they can scale to handle more messages (distributed queues). Each partition can be processed by a different set of workers. Just like databases, you'll need to specify a partition key to ensure that related messages are stored in the same partition.
5. ==**[[Backpressure]]**==: The biggest problem with queues is that they make it easy to overwhelm your system. ==If my system supports 200rps but I'm receiving 300rps, I'll **never** finish them, and the queue will continue to grow!==
	1. A queue here is just ==obscuring the real problem, which is that you don't have enough capacity!==
	2. [[Backpressure]] is a way of slowing down the production of messages when the queue is overwhelmed. This helps prevent the queue from becoming a bottleneck in your system.
		1. If a queue is full, we might want to **reject new messages** or slow down the rate at which new messages are accepted, potentially returning an error to the user or producer.

The most common queuing technologies are [[Kafka]] and [[Amazon SQS]].
- Kafka is a streaming platform that can also be used as a queue
- SQS is a fully-managed queue service provided by AWS

# Streams/Event Sourcing
- Sometimes you'll be asked a question that requires processing vast amounts of data in real-time, or supporting complex processing scenarios like [[Event Sourcing]].
	- Event Sourcing is a technique where changes in application state are stored as sequences of events, which can be replayed to reconstruct the application's state at any point in time. This is ==an effective strategy for systems that require a detailed audit trail, or the ability to reverse or replay transactions==. 
- In either case, you'll want to use a stream!
- ==UNLIKE MESSAGE QUEUES,== **Streams** can ==retain data for a configurable period of time, allowing (different) consumers to read and re-read messages from the same position, or from a specified time in the past.==
- Streams are a good choice when:
	- **You need to process large amounts of data in ==real-time==.**
		- A social media platform where you need to display real-time analytics of user engagement (likes, comments, shares) on posts. You can use a stream to ingest high volumes of engagement events generated by users across the globe. A stream processing system (like Apache [[Flink]] or Apache [[Spark]] Streaming) can process these events in real-time to update the analytics dashboard.
	- **When you need to support complex processing scenarios like ==event sourcing==.**
		- Consider a banking system where every transaction (deposit, withdrawal, transfer) needs to be recorded, and could affect multiple accounts.
			- (Note that a [[Slowly-Changing Dimensions]] solution to maintaining a version history isn't enough for this scenario; we don't just want to know the series of states that our account was in, but also the specific operations that were taken against our account (and metadata around those operations)).
	- **When you need to support ==multiple consumers== reading from the same stream**.
		- In a real-time chat application, when a user sends a message, it's published to a stream associated with the chat room. This stream acts as a centralized channel where all chat participants are subscribers. As the message is distributed through the stream, each participant (consumer) can receive the message simultaneously, allowing for real-time communication. This [[Publish-Subscribe]] pattern (Pubsub) is a common use case for streams.

Things you should know about streams for your interview:
1. **Scaling with [[Partition|Partitioning]]:** In order to scale ==streams, they can be partitioned across multiple servers==. Each partition can be processed by a different consumer, allowing for ==horizontal scaling==. 
	1. Just like with databases, you need to specify a [[Partition Key]] to ==ensure that related events are stored in the same partition==.
2. **Multiple [[Consumer Group]]s**: Streams can support multiple consumer groups, allowing different consumers to read from the same stream independently. This is useful for scenarios where you need to process the same data in a different way.
	1. In a real-time analytics system, one consumer group might process events to update a dashboard, while another consumer group might process the same events to store them in a database for historical analysis.
3. **[[Replication]]:** To support fault tolerance, streams can replicate data across multiple servers, ensuring that if a server fails, the data can still be read from another server. Combined with partitioning, we might have something like:
	1. Machine 1: (MyStream Partition 1 Leader, MyStream Partition 2 Replica)
	2. Machine 2: (MyStream Partition 2 Leader, MyStream Partition 1 Replica)
4. **[[Windowing]]:** Streams can support windowing, which is a ==way of grouping events together based on time or count==. This is ==useful for scenarios where you need to process events in batches, such as calculating hourly or daily aggregates of data.== Imagine a real-time dashboard that shows mean-delivery-time per region over the last 24 hours.

The most common streaming technologies are [[Kafka]], [[Flink]], and [[Kinesis]].

# Distributed Lock
- For a system like **TicketMaster**, you might need a way to lock some resource, like a concert ticket, for a short time (~10 minutes), so that while a user is buying a ticket, no one can steal it from them.
- Traditional databases with ACID properties use transaction locks to keep data consistent, but they're not designed for longer term locking. This is where distributed locks come in handy!
	- (("They're not designed for longer term locking" needs some more explanation, I think.))
	- ((Comments: "Distributed locks are higher-level global, often business-case-oriented locks. The pessimistic locks we discussed earlier are transaction-level locks, for doing specific updates to specific tables. When you scale to multiple machines, you need distributed locks -- the other doesn't cut it!))
- ==Distributed locks== are perfect for situations where you need to ==lock something across different systems or processes== for a reasonable period of time.
	- Often implemented using a distributed KV store like [[Redis]] or [[ZooKeeper]]. 
		- Basic Idea: ==Use a KV store to store a lock, and use the atomicity of the KV store to ensure that only one process can acquire the lock at a time.==
		- **==Example==:** If you have a Redis instance with key `ticket-123` and you want to lock it, you can set the value of `ticket-13` to `locked`. If another process tries to set the value to locked, it will fail, because the value is already set to locked. Once the first process is done, it can set the value of `ticket-123` to unlocked, and another process can acquire the lock.
			- ((I believe this requires an atomic compare-and-swap operation, maybe? I feel like I remember this))
- Distributed locks can handily be set to ==expire== after a certain amount of time, which is great for ensuring that these locks don't get stuck in a `locked` state if a process crashes or is killed.
- Use Cases of distributed locks:
	- **E-Commerce Checkout Systems**: Use a lock to hold a high-demand item, like limited-edition sneakers, in a user's cart for a short duration (like 10 minutes) during checkout to ensure that while one user is completing the payment process, the item isn't sold to someone else.
	- **==Ride-Sharing Matchmaking==**: Used to manage the assignment of drivers to riders. When a rider requests a ride, the system can lock a nearby driver, preventing them from being matched with other riders simultaneously. This lock can be held until the driver confirms or declines the ride, or until a certain amount of time has passed.
	- **Distributed Cron Jobs**: Systems that run scheduled tasks (cron jobs) across multiple servers, a distributed lock ensures that a task is executed by only one server at a time. For instance, in a data analytics platform, a daily job aggregates user data for reports. A distributed lock can prevent the duplication of this task across multiple servers to save compute resources.
	- **Online Auction-Bidding Systems**: In an online auction, we can use a distributed lock during the final moments of bidding to ensure that when a bid is placed in the last seconds, the system locks the item briefly to process the bid and update the current highest bid, preventing other users from placing a bid on the same item simultaneously.
		- ((It seems like this is a way of linearizing the history of writes... but does this require a distributed lock? Hmmm, maybe...))

Things to know about distributed locks for your interview:
- **Locking Mechanisms**: There are different ways to implement distributed locks. One common implementation uses Redis, and is called ==Redlock==. 
- **Lock Expiration**: Distributed locks can be set to expire after a certain amount of time, which is important for ensuring that locks don't get stuck if a process crashes or is killed.
- **Locking Granularity**: Distributed locks can be used to lock a single resource or a group of resources; You might want to lock a single ticket in a ticketing system, or you might want to lock a group of tickets in a section of a stadium.
- **[[Deadlock]]s**:  Deadlocks can occur when two or more processes are waiting for eachother to release a lock. ==You should be prepared to discuss how to prevent this, if it's applicable to your problem..==


# Distributed Cache
- In most system design interviews, you'll be asked to both scale your system and lower system latency. A common way to do both of these is to use ==distributed caches==, which are just a (cluster of) server(s) that stores data in memory!
	- ==Caches are great for storing data that's expensive to compute or expensive to retrieve from a database.==
- You'll want to use a cache to:
	- ==**Save aggregated metrics**==: Consider an analytics platform that aggregates data from numerous sources to display on a dashboard. The data for these ==metrics is expensive to compute==, so we don't want to recompute them every time someone requests the dashboard. Instead, we calculate these metrics ==asynchronously== (e.g. hourly, on a background job) and store the results in a distributed ==cache==. When a user requests a dashboard, the platform can retrieve the data from the cache instead of recomputing it, reducing latency!
	- ==**Reduce Number of DB Queries**==: In a web app, ==user sessions== are often stored in a distributed cache to reduce the load on the database. This is important for systems with a large number of concurrent users. When a user logs in, the system can store their session data in the cache, letting the system quickly retrieve the data when the user makes a request.
	- ==**Speed up Expensive Queries**==: Some ==complex queries== take a long time to run on a traditional, disk-based database. For example, if you have a social media platform like twitter, you might want to show users a list of posts from people they follow! This is a complex query that requires joining multiple tables and filtering by multiple columns, which could take a while. Instead, you can ==run the query once, store the results in a cache, and retrieve the results from when a user requests them==.

Things you should know about distributed caches:
1. **==[[Eviction Policy]]==**: Distributed caches have different eviction policies that determine what items are removed on the occasion ==when the cache is full==. Common ones are **Least Recently Used (LRU)**, **First-In First-Out (FIFO)**, and **Least Frequently Used (LFU)**.   
2. ==**[[Cache Invalidation Strategy]]==**: This is the strategy you'll use to ensure that the data in the cache is up to date. If you're designing TicketMaster and caching popular events, you'll need to **invalidate** the cache (manually) if the event in your Database was updated (like the venue changed; you'll want to invalidate the existing cache entry for that event!)
3. ==**[[Cache Write Strategy]]==**: This is the strategy used to make sure that data is written to your cache in a consistent way. Strategies include:
	1. ==[[Write-Through Cache]]:== **Writes data to BOTH the cache and the underlying datastore** **simultaneously**! Ensures that your cache is **consistent** with your backing datastore, but can be **slower** for write operations.
	2. ==[[Write-Around Cache]]:== **Writes data directly to the datastore, bypassing the cache**. This can minimize **==[[Cache Pollution]]==** but might increase data fetch times on subsequent reads when you have a cache miss. 
		1. ((It makes sense to me that you wouldn't want to use this if your items are uniformly read, but rather if you have a lopsided distribution of reads. It doesn't make sense to write an item to the cache if it's never going to be requested, or if it's more important to prioritize the response time of more common requests, which will naturally populate the database.))
	3. ==[[Write-Back Cache]]==: **Writes data to the cache and then asynchronously writes the data to the datastore**. This can be faster for write operations but can lead to ==**data loss**== if the cache is not persisted to disk.
		1. ((This seems pretty dumb unless you really need to use it. I know that some in-memory datastores do have the ability to async persist to disk so you might not lose all data, but... Yuck.))

==**TIP:**== Don't forget to be explicit about what data you're storing the cache, including the data structure you're using. Modern caches have many different data structures that can be leveraged, and are not just simple key-value stores. So if you're storing a list of events in your cache, you might want to use a ==sorted set== so that you can easily retrieve the most popular events. =="I'll store the events in a cache" might be fine, but can be a missed opportunity to share knowledge.==

Most common distributed caches:
- [[Redis]] and [[Memcached]] are the most popular.
	- Redis is a KV store supporting many different datastructures, including strings, hashes, lists, sets, sorted sets, bitmaps, and hyperloglogs. 
	- Memcached is a simple Key-Value store supporting strings and binary objects.

# CDN
- What's a CDN and when should be use it?
- Modern systems often serve users globally, making it hard to deliver content quickly to users all over the world -- [[Content Distribution Network|CDN]]s are types of ==caches== that use ==distributed servers== to deliver content to users based on their geographic location.
	- ==Often used to deliver **static content** such as images, videos, and HTML files, but can also be used to deliver dynamic content like API responses.==
	- Works by caching content on servers that are close to users. When a user requests content, the CDN routes the request to the closest server. If the content is cached on that server, the CDN will return the cached content. If the content is NOT cached on that server, the CDN will fetch the content from the origin server, cache it, and return the content to the user.
- The most common use case in interviews is to use CDNs to cache static media assets like images, videos, and Javascript files which are often stored in object stores).

Things to know about CDNS
1. **CDNs are not just for static assets!** They can be used to cache dynamic content; especially useful for content that's addressed frequently, but changes infrequently. A blog post that's updated once a day can be cached by a CDN, for instance.
2. **CDNs can be used to cache API responses!** If you have an API response that's accessed frequently, you can use a CDN to cache the responses. This can help reduce the load on your servers and improve the performance of your API!
3. **Eviction policies**. Like other caches, CDNs have eviction policies that determine when cached content is removed. For example, you can set a **TTL** for cached content, or you can use a **cache invalidation mechanism to remove content from the cache when it changes** ((for example if you have something like a blog post that you've updated, you could manually trigger that invalidation as part of your update flow.)).

Examples of CDNs
- Some of the most popular CDNs are Cloudflare, Akamai, and [[Amazon CloudFront]]. 
- These CDNs offer ar ange of features, including caching, DDoS protection, and web application firewalls.
- These also have a global network of edge locations, which means that they can deliver content to users around teh world with low latency.