System Design In a Hurry: https://www.hellointerview.com/learn/system-design/in-a-hurry/core-concepts

_________________ 

![[Pasted image 20250518115352.png|500]]

# Scaling
- One of the most important topics is how to deal with scale, in a SD interview.
- [[Horizontal Scaling]] is about adding **more machines** to a system to increase its capacity.
	- Adding machines isn't a free lunch! Oftentimes by scaling, you're forced to contend with the distribution of work, data, and state across your system.
	- Inexperienced candidates tend to make two mistakes:
		- ==Mistake==: Leaping to horizontal scaling to solve any performance problem, even if it's not necessary.
		- ==Mistake==: Not considering the implications of horizontal scaling on the rest of the system.
	- You'll need to ==consider how to distribute work across your machines.== Most modern systems use a technique called [[Consistent Hashing]] to distribute work across a set of machines, a technique that arranges both data and machines in a circular space called a hash ring, allowing us to add or remove machines with **minimal data redistribution**.
- [[Vertical Scaling]] is the process of adding **more resources to a single machine(s)** to increase its capacity.
	- Most senior interviewers will recognize that ==vertical scaling actually requires significantly less incremental complexity==!
	- ==If you can estimate your workload== and determine that you can scale vertically for the foreseeable future, it's often a better solution than horizontal scaling.

#### Work Distribution
- The first challenge of [[Horizontal Scaling]] is getting the work to the right machine. 
- This is often done via a [[Load Balancing|Load Balancer]], which will ==choose which node from a group to use for an incoming request.== While load balancers often come with many different ==load balancing strategies== (e.g. **least connections**, **utilization-based**), simple **round-robin** allocation is often sufficient.
	- For asynchronous jobs work, this distribution is often done via a queue system.
- ==The goal of work distribution strategies== is to **keep the load on the system as even as possible**. If you're using a hash map to distribute work across a set of nodes, you might find that one node is getting a disproportionate amount of work because of the distribution of incoming requests.
	- Your scalability depends on how well you can distribute work! If one node is 99% busy and the remaining are 10% busy, you're not getting much out of your horizontal scaling!

#### Data Distribution
- We also need to consider how to distribute data across the system.
- Most frequently, this implies keeping data in a database that's shared across all nodes that need access to it.
	- At other times, the nodes processing requests have to keep data in-memory on the same node!
- ==Ideally, we'd like for a single node to access the data it needs without needing to talk to another node.== 
	- If you need to talk to other nodes (a concept known as [[Fan-Out]]), try to keep the number small.
	- A ==common antipattern== is to have requests that fan out to many different nodes, then gather the results together.
		- This [[Scatter-Gather]] pattern can be ==problematic== because it leads to a lot of network traffic, and more importantly is sensitive to failures in each connection, suffering from [[Tail Latency]] issues if the final result is dependent on every response.
- ==NOTE:== If your system design problem involves Geography, there's a good chance that you have the chance to partition by some sort of `REGION_ID`. 
	- For systems that involve physical locations and the user is only concerned in data around a particular location (e.g. Yelp, where a user in the US doesn't need to know about data in Europe), you this can be a great way to scale, assuming all of your traffic isn't in the United States.
- Inherently, [[Horizontal Scaling]] introduces ==synchronization challenges!==
	- You're either:
		- Reading and writing to a shared database, which is a network hop away (~1-10ms, ideally)
		- Keeping multiple redundant copies across each of your servers.
	- These can both result in race conditions and consistency challenges.
	- Most database problems are built to resolve some of these directly, e.g. using [[Transaction]], but in other cases you might need to use a [[Distributed Lock]]. 
		- Regardless, you'll have to be prepared to talk about how you're going to keep your data [[Consistency|Consistent]].


# CAP Theorem
- The [[CAP Theorem]] is a fundamental distributed system concept saying that you can only have two of three properties: [[Consistency]], [[Availability]], and [[Partition]] Tolerance. In practice, since network partitions are unavoidable, this means ==choosing between consistency and availability== in the face of these unavoidable network partitions.
- Choosing ==Consistency== means that all nodes in your system see the same data at the same time.
	- When a write occurs, all subsequent reads will return that value, regardless of which node they hit.
	- During a network partition, some nodes may become unavailable to maintain this consistency guarantee.
- Choosing ==Availability== means that every request will receive a response, even during a network partition. 
	- The tradeoff is that different nodes may "temporarily" have different versions of the data, leading to inconsistency. 
	- The system will *eventually* reconcile these differences, but here's no strong guarantee about when this will happen.

- ==In a system design interview, **[[Availability]] should be your default choice.**==
	- You only need [[Strong Consistency]] in systems where reading stale data in unacceptable, such as:
		- ==Inventory management systems==, where stock levels need to be precisely tracked to avoid overselling products.
		- ==Booking systems for limited resources== (airline seats, event tickets, hotel rooms) where you need to prevent double-booking.
		- ==Banking systems== where the balance of an account must be consistent across all nodes to prevent fraud.

# Locking
- In our system, we may have shared resources which can be accessed by one client at a time.
	- An example might be a shared counter (inventory units) or an interface to a physical device (drawbridge up!).
- [[Locking]] is the process of ensuring that ==only one client can access a shared resource at a time.==
	- Locks happen at every scale of computer systems (OS kernels, databases, distributed locks).
- ==In most system design interviews, you'll be forced to content with locks when considering [[Race Condition]]s==, which are situations where multiple clients are trying to access the same resource at the same time (if improperly handled, can lead to data corruption, lost updates, other bad things).
- There's **three things to worry about when employing locks:**
	- **Granularity of the lock**:
		- ==Locks should be as fine-grained as possible!== We want to lock as **little as possible** to ensure that we're not blocking other clients from accessing the system!
			- If we're updating a user's profile, we want to lock only that user's profile, and not the entire user table!
	- **Duration of the lock**
		- ==Locks should be held for as short a time as possible!== We want to lock only for the duration of the critical section. 
			- If we're updating a user's profile, we want to lock only for the duration of the update, and not for the entire request.
	- **Whether we can bypass the lock**:
		- In many cases, we can ==AVOID LOCKING== by employing an [[Optimistic Concurrency Control]] (OCC) strategy, especially if the work to be done is either read-only or can be retired.
		- In an optimistic strategy, ==we're going to assume that we can do the work without locking, and then check to see if we were right. In most systems, we can use a "compare and swap" operations to do this.==  
		- **OCC** is useful when most of the time we don't have contention in the system (which is a good assumption for most systems!).  But it's not a good match for all systems (e.g. updating a bank user's balance).

![[Pasted image 20250518123007.png|700]]

# Indexing
- [[Indexing]] is about making data faster to query.
- ==In many systems, we can tolerate slow writes, but we can't tolerate slow reads.== (On the logic that most of a user's operations are going to be read operations, with few write operations, which is common).
- The most basic method of indexing is simply keeping our data in a [[Hash Map]] by a specific key, such that we can grab the data we need in O(1) time, rather than needing to scan our dataset to find the record we need.
- Another way is to keep our data in a sorted list, which allows us to do a binary search to find the data we need in O(LogN) time; This is a common way of indexing data in databases.
- There are many different types of indexes, but the principle is the same: Doing a minimal amount of extra up-front work so that your reads can be extra fast.

#### Database Indexes
- Most questions regarding indexing will happen inside your database! ==Your database choice will impact eh options that you have for indexing.==
	- In most relational databases, you can create indexes on any column or group of columns in a table.
	- Databases like DynamoDB allow you to create many secondary indexes, and others like Redis leave you on your own to design and implement your own indexing strategy.

#### Specialized Indexes
- There are many specialized indices used to solve specific problems.
- [[Geospatial Index]]es are used to index location data, useful for systems that need to do things like find the nearest restaurant or the nearest gas station.
- [[Vector Database]] are used to index high-dimensional data, useful for systems that need to do things like find similar images or similar documents.
- [[Full-Text Search Index]]es are used to index text data, useful for things that need to do things like search for documents or search for tweets.
- Many mature databases like [[PostgreSQL|Postgres]] support extensions that allow you to create specialized indices.
	- For examples, Postgres has a ==PostGIS== extension that allows you to create geospatial indexes. 
- If not, you'll need to maintain your indexes externally.
	- [[ElasticSearch]] is our recommended solution for these secondary indexes, when it can work. It supports full-text indexes via [[Lucene]], geospatial indexes, and even vector indexes.
	- You can set ElasticSearch up to index most databases via [[Change Data Capture]] (CDC), where the ES cluster is listening to changes coming from the database, and updating its indexes accordingly.
	- Still, ==having these secondary external systems storing indexes that your primary database might support isn't a perfect solution; you introduce a new point of failure and a new source of latency, along with potential consistency considerations!==

# Communication Protocols
- Protocols are an important part of software engineering, but most system design interviews don't cover the full OSI model. Instead, you'll be asked to reason about the communication protocols used to build your system.
- There are ==two categories of protocols to handle:== **internal** and **external**.
- For 90%+ of system design problems and typical microservice applications, either [[HTTP]](S) ((?? Do they mean [[REST]]?)) or [[gRPC]] will do the job -- don't make things complicated.
- Externally, you'll need to ==consider how clients will communicate with your system:==
	- **Who initiates** the communication.
	- What are the **latency considerations**.
	- **How much data** needs to be sent.
- Across most choices, systems can be built with a combination of [[REST]], [[Server-Sent Event]] (SSE) or [[Long Polling]], and [[Websockets]].

![[Pasted image 20250518130221.png|700]]

- Use HTTP(S) for APIs with simple request and responses. 
	- Because each request is stateless, you can scale your API horizontally by placing it behind a load balancer.
- If you need to give your client near-realtime updates, you need a way for clients to receive updates from the server.
	- [[Long Polling]] is a great way to do this that blends the simplicity and scalability of HTTP with the realtime updates of Websockets. With Long Polling, ==clients make a request o the server, and the server holds the request open until it has new data to send to the client.== Once the data is sent, the client remakes *another* request and the process repeats.
		- ==BONUS==: You can use standard load balancers and firewalls with long polling -- no special infrastructure needed.
- [[Websockets]] are necessary if you need realtime, bidirectional communication between the client and server.
	- From a system design perspective, websockets can be ==challenging because you need to maintain the  connection between client and server==. This can be challenging for load balancers and firewalls, and it can be a challenge for your server to maintain many open connections.
		- **==NOTE==**: A common pattern is to use a **message brokers** to handle the communication between the client and the server, and for the backend services to communicate with this message broker. This ensures that you don't need to maintain long connections to every service in your backend.
- [[Server-Sent Event]]s (SSE) are a great way to send updates from the server to the client.
	- Similar to long polling, but more efficient for unidirectional communication from the server to the client.
	- ==SSE allows the server to push updates to the client whenever new data is available, without the client having to make repeated requests as in long polling==.
	- This is achieved through a ==single, long-lived HTTP connection==, making it more suitable for scenarios where the server frequently updates data that needs to be sent to the client.
	- Unlike WebSockets, SSE is designed specifically for server-to-client communication and **does not support client-to-server messaging.** 
	- **This makes SSE simpler to implement and integrate into existing HTTP infrastructure, such as load balancers and firewalls, without the need for special handling.**

COMMENT: ==Statefulness== is a major source of complexity for systems. Where possible, relegating state to a message broker or database is a great way to simplify your system. This enables your services to be stateless and horizontally scalable while still maintaining stateful communication with your clients.

# Security
- When designing production systems, security should be top of mind!
- Rarely will system designs require you to do detailed security testing of your design, but they're looking for you to emphasize security where appropriate.

##### Authentication/Authorization
- In many systems, you'll expose an API to external users which needs to be locked down to only specific users.
- Delegating this work to either an [[API Gateway]] or a dedicated service like [[Auth0]] is a great way to ensure that you're not reinventing the wheel.
- ==It's often sufficient for you to say to your interviewer: "My API Gateway will handle authentication and authorization."==
##### Encryption
- Once you're handling sensitive data, it's important to make sure you're keeping it from snooping eyes.
- You'll want to cover both:
	- **Data in ==Transit**==: Using protocol encryption; HTTPS is the [[Transport Layer Security]] protocol that encrypts data in transit and is the standard for web traffic. [[gRPC]] supports SSL/TLS out of the box.
	- **Data at ==Rest**==: You'll want to use a database that supports encryption, or **otherwise encrypt the data yourself before storing it**!

**==COMMENT==**: For sensitive data, it can often be useful for the end-user to control the keys. This is a common pattern in systems that need to store sensitive data. For example, if you're building a system that stores user data, you might want to encrypt that data with a key that's unique to each user. That way, even if your database is compromised, the data is secure.


# Monitoring
- Once you've designed your system, some interviewers might ask you to discuss how you'll monitor it. 
	- The idea here is simple: candidates that understand monitoring are more likely to have experience with actual systems in production.
	- Monitoring real systems is also a great way to learn about how systems actually scale (and break).
- ==**Monitoring generally occurs at three levels, and it's useful to name them!**==
	- ==**Infrastructure Monitoring**==
		- The process of monitoring the health and performance of your infrastructure. Includes things like ==CPU usage, memory usage, disk usage, and network usage==. This is done with a tool like **Datadog** or **New Relic**.
		- A disk usage alarm may not break down your service, but it's usually a leading indicator of problems that need to be addressed.
	- ==**Service-Level Monitoring**==
		- The process of monitoring the health and performance of your services.
		- Includes things like ==request latency, error rates, and throughput==. 
			- If your service is taking too long to respond to requests, it's likely that your users are having a bad time.
			- If your throughput is spiking, it may be that you're handling more traffic, or your system might be misbehaving.
	- ==**Application-Level Monitoring**==
		- Monitoring the health and performance of your application.
		- This includes things like ==number of users, number of active sessions, and number of active connections==. 
		- Might also include ==key business metrics for the business.==
		- This is often done using a tool like **Google Analaytics** or **Mixpanel**
