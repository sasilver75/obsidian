See: https://www.hellointerview.com/learn/system-design/in-a-hurry/patterns

--------
![[Pasted image 20250518175647.png|400]]
What follows are some common patterns that you can use to build systems; you'll often find yourself combining them to build a system.

# Simple DB-Backed CRUD Service with Caching
- The most common pattern is a simple **CRUD service** that uses a **database for storage** and **caching to improve performance**. You'll also use a **load balancer** to distribute traffic across multiple instances of your service, and an **API gateway** in front of the client.
	- ==This is the pattern you'll use for most web applications.==
![[Pasted image 20250519121658.png|450]]
- Many designs start with this pattern, then add additional complexity as requirements are added.
	- e.g. You might start with a simple CRUD service and then add a **search index** to improve search performance.
	- e.g. You might start with a simple CRUD service and then add a **queue** to handle async processing!

**NOTE:** This design is "==too simple==" for all but the most junior roles, ==but it's a good start==! For all levels, we recommend moving quickly in your interview to solve requirements so that you can spend ample time optimizing your design and handling deep dives. ==Pound it out and move on to the next thing.==

# Async Job Worker Pool
- If you have a system which needs to handle a lot of processing and can tolerate some delay, you might use an async job worker pool. 
	- ==This async job pattern is common in systems that need to process a lot of data, like a social network that needs to process many images or videos.==
	- You'll use a **queue** to store jobs, and a **pool of workers** to process those jobs.
- A popular option for the queue is [[Amazon SQS]], and for the workers, you might use a pool of [[Amazon EC2]] or [[Amazon Lambda]] functions.
	- SQS guarantees ==**at-least-once delivery**== of messages, and the workers will respond back to the queue with heartbeat messages to indicate that they're still processing the job.
		- If a worker fails to respond with a heartbeat, the job will be retried on another host.
- Another option is for your queue to instead be a log of events stored on something like [[Kafka]], which gives you many of the same guarantees as SQS, but since requests are written to an append-only log, you can always **replay** the log to reprocess events if something goes wrong.
![[Pasted image 20250519122203.png|250]]



# Two Stage Architecture
- A common problem in system design is to "scale" an algorithm with poor performance characteristics.
- Consider the problem of trying to find two images that look about the same
	- There are many available algorithms that can compare the *two images* and give a score for similarity
	- However these are wholly inappropriate for a system that needs to compare a *large number* (e.g. thousands) of images!
	- So what can we do? We use a ==two stage architecture==
		- In the **first stage:** we use a fast algorithm to filter out the vast majority of dissimilar images.
		- In the **second stage**: We use a slower algorithm to compare the remaining images.
- This architecture occurs in [[Recommendation System]] (e.g. candidate generators), [[Search Engine]]s (e.g. [[Inverted Index]]es), route planning (e.g. ETA services), any many other systems.
![[Pasted image 20250519122531.png|300]]
Above: a two-stage architecture for image search ((It's not clear to me what they're trying to communicate, but I imagine they're doing some sort of ANN-type search using their vector database and then doing some more expensive reranking and selection in their ranking service.))

# Event-Driven Architecture
- An [[Event-Driven Architecture]] (EDA) is a design pattern centered around events!
- This architecture is useful in systems where it's CRUCIAL to react to changes in real-time! ==EDA helps in building systems that are **highly-responsive**, **scalable**, and **loosely-coupled**==.
- The core components are:
	- **Event producers**: Generate a stream of events that are sent to an event router.
	- **Event routers** (or brokers): e.g. [[Kafka]] or [[AWS EventBridge]] **dispatch** these events to appropriate consumers based on the event type or event content.
	- **Event consumers**: Process the events and take necessary actions, which could range from sending notifications to updating databases or triggering other processes.
- **NOTE**: One of the more important design decisions in event-driven architectures is ==how to handle failures when they occur!== 
	- Brokers like [[Kafka]] keep a durable log of their events with configurable retention, which allows processors to pick up where they left off! **This can be a double-edged sword**! 
	- If you system can only process N messages per second, you might quickly find yourself in a situation where you'll take hours or days to catch back up, with the service substantially degraded the entire time. Be careful about where this is used!
- Example use
	- **E-Commerce** system where an event is emitted each time an order is placed.
		- This can simultaneously trigger multiple downstream processes, like **order processing**, **inventory management**, and **notification systems** simultaneously.

This pattern supports flexibility in system interactions, and can **easily adapt to changes in process or business requirements!** It can significantly enhance the system's ability to handle high loads and facilitate complex workflows.

# Durable Job Processing
- Some systems need to manage **long-running jobs** that can take **hours or days to complete**.
	- If the system crashes, **==we don't want to lose the progress of the job!==** 
	- Ideally, ==we'd also like to be able to scale the job across multiple machines.==
- A common pattern:
	- Use [[Kafka]] to store the jobs, and then have a pool of workers that can process the jobs.
	- The workers periodically checkpoint their progress to the log, and if a worker crashes, another worker can pick up the job where the last worker left off.
- Another option is to use a technology like [[Temporal]].

**WARN:** Setups like this can be difficult to evolve with time... if we want to change the format of the job, we'll need to handle both the old and new formats for a while.
![[Pasted image 20250519124001.png|400]]
Above: Managing ==multi-phase jobs== using a distributed, durable log (e.g. Kafka)

# Proximity-Based Services
- Several systems like Uber or GoPuff will require you to search for entities by location!
- [[Geospatial Index]]es are the key to being able to efficiently query and retrieve entities based on their geographic proximity. 
- **Options:** These services often rely on extensions to commodity databases like [[PostgreSQL|Postgres]] with [[PostGIS]] extensions, or on [[Redis]]'s geospatial datatype, or dedicated solutions like [[ElasticSearch]] with geo-queries enabled.
- These architectures typically involve dividing **up** the geographic area into manageable regions and indexing entities within these regions. This allows the system to quickly exclude vast areas that don't contain relevant entities, thereby reducing the search space significantly.
- ==NOTE==: Geospatial indices are great, but if you've only got 1,000 items, you may be better off just scanning all of the items rather than incur the overhead of managing an additional purpose-built index or service.
- **==NOTE==**: Most systems won't require users to be querying *globally*; often, when proximity is involved, users are just looking for entities *local* to them!****