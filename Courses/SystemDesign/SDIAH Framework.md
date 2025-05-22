System Design In a Hurry: https://www.hellointerview.com/learn/system-design/in-a-hurry/introduction

_________________ 

# Delivery Framework

- The easiest way to SABOTAGE your chances is to fail to deliver a working system.
- ==Failing to deliver a working system is the most common reason that mid-level candidates fail these interviews, and it often manifests as the opaque "time management."==
	- It's not that you need to work twice as fast, you often just need to focus on the right things.
- By structuring your interview in this way, you'll stay focused on the bits that are important to your interviewer, and you'll have a clear path to fallback if you're overwhelmed and nervous.

![[Pasted image 20250510170635.png]]
## 1) ==Requirements== (~5 Minutes)
The goal is to get a clear understanding of the system that you're being asked to design. 
We suggest that you break you requirements into two sections
##### 1A) ==Functional Requirements== ("users should be able to...")
- These are the core features of your system.
- Often times a back and forth with your interviewer: "Does the system need to do X? What would happen if Y?"
- For a system like Twitter, you might arrive at:
	- Users should be able to post tweets
	- Users should be able to follow other users
	- Users should be able to see tweets from users they follow
- For a system like a Cache, it might be:
	- Clients should be able to insert items
	- Clients should be able to set expirations
	- Clients should be able to read items
- ==Keep your requirements targeted, and prioritize the most important ones! Many systems will have hundreds of features, but it's your job to identify and prioritize the top three. Having a long list will hurt more than help.==
##### 1B) ==Non-Functional Requirements== ("the system should be...", "the system should be able to...")
- Non-functional requirements are statements about the system qualities that are important to your users.
- For a system like Twitter:
	- The system should be highly available, prioritizing availability over consistency
	- The system should be able to scale to support 100m+ DAUs
	- The system should be low-latency, rendering feeds in under 200ms
		- ==Important:== Try to *quantify* where possible! "low-latency" is obvious and not very meaningful; "Search must be <500ms" is more useful as it identifies the part of the system that most needs low-latency, and provides a target.

- Coming up with non-functional requirements can be challenging, but here are some things to consider that might help you identify important non-functional requirements:
	1. **==CAP Theorem==**: Should your system prioritize ==consistency== or ==availability==? Partition-tolerance is a given in distributed systems, so we really have to choose either consistency or availability.
	2. **==Environment Constraints==**: Are there any constraints on the environment in which your system will run? Are you running on a mobile device with limited battery life? Running on devices with limited ==memory/bandwidth==? 
	3. **==Scalability==**: All systems need to scale, but does this system have unique scaling requirements? Will it have bursty traffic at a specific time of day? Are there ==special events== like holidays that will cause a significant increase in traffic? Also consider read vs. write ratio: Does your system need to scale ==reads or writes more==?
	4. **==Latency==**: How quickly does the system need to respond to user requirements? Specifically consider any requests that might require ==meaningful computation==, for example low-latency search when designing Yelp.
	5. **==Durability==**: How important is it that the data in your system is not lost? A social network might be able to tolerate some data loss, but a banking system cannot.
	6. ==Security==: How secure does the system need to be? Consider data protection, access control, and compliance with regulations.
	7. ==**Fault Tolerance**==: How well does the system need to handle failures? Consider ==redundancy==, ==failover==, and ==recovery== mechanisms.
	8. **==Compliance==**: Are there ==legal or regulatory requirements== that the system needs to meet? Consider industry standards, data protection laws, and other regulations.
##### 1C) Capacity Estimation
- Many guides will suggest doing back-of-envelope calculations at this stage.
- We believe ==these back-of-the-envelope are often unnecessary! Only perform calculations if they directly influence your design==
	- Many candidates will calculate storage, DAU, and QPS, only to conclude "Ok yeah, so it's a lot. Got it."
		- Interviewers don't get anything from this, except that you can perform basic arithmetic.
- ==Explain to the interviewer that you'd like to skip on estimations upfront and that we'll do math when designing when/if necessary.==
	- It might be necessary if we're designing a TopK system for trending topics on facebook posts, where we'd first want to estimate the number of topics we'd expect to see, since this would influence if we can use a single instance of a data structure like a min-heap or if we need to shard it across multiple instances, which will have a big impact on your design.

## 2) ==Core Entities== (~2 minutes)
- Next, take a moment to identify and list the core entities of your system.
- This helps you to define terms, understanding the data central to your design, and give you a foundation to build on.
- Identify the ==core entities that your API will exchange, and that your system will persist in a Data Model==.
	- This is as simple as jotting down a ==bulleted list== and explaining this to your first draft to the interviewer.
		- ==Why not list the entire data model?== Because you don't know what you don't know! As you design the system, you'll discover new entities and relationships that you didn't anticipate. 
		- By starting with a small list, we can quickly iterate and add to it as we go.
	- Once you get into the high level design and have a clearer sense of exactly what state needs to update on each request, we can build out the relevant columns/fields for each entity.
- For Twitter, our core entities may be:
	- User 
	- Tweet
	- Follow
- ==Useful questions to ask yourself:==
	- "Who are the ==actors== in the system? Are they ==overlapping==?"
	- "What are the ==nouns== or ==resources== necessary to satisfy the functional requirements?"


## 3) ==API or System Interface== (~5 Minutes)
- Before we get into the high-level design, we want to define the contract between your system and its users.
- Oftentimes, this maps directly to the functional requirements that we've already identified (e.g. "User should be able to...", but not always!
- There's a quick decision to make: RESTful API or GraphQL API?
	- **RESTful API**: Standard communication constraints of the internet. Uses GET/POST/PUT/DELETE
	- **GraphQL API**: A newer communication protocol that allows clients to specify exactly what data they want to receive from the server.
	- **Wire Protocol**: If you're communicating over websockets or raw TCP sockets, you'll want to define the wire protocol. This is the format of the data that will be sent over the network, usually in the format of messages.
- ==Don't overthink this. BIAS towards creating a REST API.== Use GraphQL only if you *really* need clients to fetch only the requested data (no over- or under-fetching). If you're going to use websockets, you want to describe the wire protocol.

For Twitter, let's choose REST and have the following endpoints (notice that our Core Entities that we defined earlier {User, Tweet, Follow} are the ones that we exchange her via the API).

```
# Post a tweet
POST /v1/tweet
body: {
	"text": string
}
- [ ] 
# Get tweet detail
GET /v1/tweet/:tweetId -> Tweet

# Follow a user
POST /v1/follow/:userId

# Get tweets
GET /v1/feed -> Tweet[]
```
Above:
- Notice that here's no userId in the POST /v1/tweet endpoint?
	- This is because we get the id of the user initiating the request from the ==authentication token== in the request header.
	- Putting sensitive information like user ids in the request body is a security risk and a mistake that many candidates make, don't be one of them!


## 4) OPTIONAL: ==Data Flow== (~5 minutes)
- For some backend systems, especially data-processing systems, it can be helpful to describe the high level sequence of actions and processes that the system performs on the inputs to produce the desired outputs.
- ==If your system doesn't involve a long sequence of actions, skip this!==
- We usually define data flow as a simple list, used to inform the high-level design in the next section:
- For a web crawler, this might look like:
	1. Fetch seed URLs
	2. Parse HTML
	3. Extract URLs
	4. Store Data
	5. Repeat


## 5) ==High-Level Design== (~10-15 Minutes)
- Now that we have a clear understanding of the requirements, entities, and API of our system, we can start to design the high-level architecture.
- ==This consists of drawing boxes and arrows to represent the different components of our system and how they interact== -- often basic building blocks like ==servers==, ==databases==, ==caches==, etc.
- The Key Technologies section will give you a good sense of the most common components you'll need to know.

- ==IMPORTANT:== ==Don't over-think it!== Your primary goal is to design an architecture that satisfies the API you've designed, and thus, the requirements that you've identified.
	- In many cases, ==you can even go through your API endpoints one by-one and build your design to sequentially satisfy each one.==

- ==IMPORTANT:== ==KEEP IT SIMPLE! ==It's ==incredibly common for candidates to start laying on complexity too early==, resulting in them never arriving at a complete solution. ==Focus on a relatively simple design== that meets cross-functional requirements.

- As you're drawing your design, you should be talking through your thought process with your interviewer. ==Be explicit about how data flows through the system and in what state for each request== (either in databases, caches, message queues, etc.).
- When the request reaches your database or persistence layer, it's a ==good time to start documenting relevant columns/fields for each verify==. You can do this directly, next to your database, visually. ==Don't worry too much about types here, the interviewer can infer and it'll just slow you down.==

![[Pasted image 20250510180528.png]]
Above: An example for our Twitter system.
- See that we've basically went through our API interface that we defined earlier, and build the required services and databases for each  API call.
- See also that when required, they're defining some of the interactions between services. I can't really see, but it looks like they might have written these in an RPC format.
- ((Sam Note: It seems de rigeur to have a service for every entity. I'm not sure if we really need this in reality, but it might make sense to do it this way for interviews.))


## 6) ==Deep Dives== (~10 Minutes)
- Our simple, high-level design of Twitter is going to be woefully inefficient when it come to fetching users' feeds!
- Now with a high-level design in place, we'll harden our design by:
	- Making sure that it meets all of our ==non-functional requirements==
	- Addresses ==edge cases==
	- Identifies and addresses ==bottlenecks==
	- Improves the design based on probes from our interviewer.
- The degree to which you're proactive in leading deep dives is a function of your seniority.
	- More junior candidates can expect the interviewer to jump in and point out places where the design could be improved.
	- More senior candidates should be able to identify these places themselves and lead the discussion.

Looking at our non-functional requirements, one of them was that our system needs to ==scale== to >100M DAU
- We could lead a discussion around ==horizontal scaling==, the ==introduction of caches==, and ==database sharding==, updating our design as we go.

One of our other ones were about needing to have feeds that were fetched with ==low-latency==
- In the case of Twitter, this is actually the most interesting problem; We could lead a discussion about ==fanout-on-read== versus ==fanout-on-write== and the use of caches.

==WARNING:== ==Make it a conversation, still!==  A common mistake is that the candidate tries to TALK OVER THEIR INTERVIEWER here. Make sure to give the interviewer room to ask questions and probe your design.



















