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
- A simple, high-level design of Twitter is going to be woefully inefficienet when it comes to fetching users' feeds. No problem! We handle this in the deep dive section.
- Here, we ==harden our design by making sure it addresses non-functional requirements, addresses edge cases, identifies and addresses issues and bottlenecks, improves design based on probes from interviewer==.
- The degree to which you're proactive here is a function of your seniority.
- Talking about horizontal scaling, introducing caches, database sharding, etc... Things like fanout-on-read vs fanout-on-write and the use of caches.
- Make sure you give your interviewer room to ask questions and probe your design.















































