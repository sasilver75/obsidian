From System Design in a Hurry: https://www.hellointerview.com/learn/system-design/deep-dives/cap-theorem


In a distributed system, you can only have 2 out of 3:
1. [[Consistency]]: All nodes/users see the same data at the same time
2. [[Availability]]: Every request gets a response (successful or not)
3. [[Partition]] Tolerance: System works despite network failures between nodes

==**WARN**==: Note that **consistency** in the context of the CAP theorem is quite different than the consistency gauranteed by [[ACID]] databases.

In an interview, we'll often first align on:
1. Functional Requirements
2. Non Functional Requirements
	1. Here, we want to ask ourself: Does this system (or some *part* of the system) need to prioritize **Consistency** or **Availability?**
		1. This will have a significant influence in our design, later in the deep dives!

But why do we need to pick either Consistency or Availablity? Why not both?
![[Pasted image 20250519162604.png]]
Say that we have an example above:
- We have a hosted website with two servers: One hosted in the USA, one in Europe
- An American writes to the American Server, updating their name in the profile. 
- This data is replicated to the server in Europe so that when a European asks for that information, they get the most recent information.
- What happens if the network connection (replicate) goes down before we had a chance to replicate the information from the American server to the European server?
	- Should we give the European user an error, because we don't want them to view the stale data?
		- If so, we've prioritized [[Strong Consistency]].
		- This is what we'd want to do for scenarios where we can't tolerate discrepancies in visible data, such:
			- as booking systems for airlines, hotels, etc. We wouldn't want someone to see that Seat 6A is still available when it's really not (and worse, be able to actually *book* it!)
			- or Ecommerce scenarios where there's one toothbrush left and both people think that the they've bought it!
	- Should we give the European user the stale data?
		- If so, we've prioritized [[Availability]].
		- There are times where this is totally fine!
			- Social media App
			- Yelp-like business review service (it might be fine that a Yelp user sees slightly outdated business information for a few seconds, rather than not showing them the restaurant!)
			- Netflix: If we change the description on a new movie, or add a new movie, is it okay that a user in Canada doesn't see 

So ask yourself when you're going over the Non-Functional Requirements:
- =="**Does it matter that every single user sees the same state of the system at any given time, and if they didn't, would it be catastrophic?**"==
	- If it would be catastrophic, you prioritize [[Strong Consistency]]
		- Might need to implement things like [[Distributed Transaction]]s
			- When a write happens to a cache and a database (write through), you want to make sure that it happen to both atomically!
		- Might need to limit certain things to a single node
			- e.g. if you Database is a single instance, you can't have these propagation issues!  For you airline system, you might have a single database (e.g. Postgres) for which we can issue atomic transactions.
		- Might need to accept higher latency (showing users Spinners as we wait for propagation to happen between instances)
		- Example tools:
			- PostgreSQL
			- Trad RDMS
			- Spanner
			- NoSQL databases that have strong consistency modes (e.g. DynamoDB)
	- If it doesn't matter, you can prioritize [[Availability]]!
		- In this case, you can use multiple replicas (where eventual consistency is used)
		- [[Change Data Capture]] (CDC) is okay to use in your system (which is necessarily eventually consistent)
		- Example tools
			- DynamoDB (even in Multi-AZ Mode)
			- Cassandra


For Advanced Candidates:
- ==**Different parts of a system might have different requirements!==**
	- Ticketmaster
		- Availability for CRUD on events
			- It's okay if event descriptions being updated are eventually consistent; it's better that people can always view the event, because that's what makes us money!
		- Consistency for booking tickets
			- However for actually creating the tickets, let's prioritize consistency, so that we don't have people double-booking!
	- Tinder
		- Availability for viewing profile data
			- It's okay if people see my old picture for a while if I change my profile picture.
		- Consistency for matching
			- If user A swiped on me in Europe, and then I swipe on them in the US, I want to IMMEDIATELY show them a Match "You Matched" if you swipe, because that's a big product feature; so we want to have a consistent view of matches.

You'll hear ==Consistency== being used in the context of the CAP Theorem, but there's actually varying levels of it:
1. [[Strong Consistency]]: All reads reflect the most recent write; everyone sees the same data.
2. [[Causal Consistency]]: Related events appear in order. You won't see a reply to a comment before you see the comment itself.
3. [[Read-your-Writes Consistency]]: I as a user should have a consistent view of what I'm just done (I should see my own profile picture change, or I'll think it's broken, but the system doesn't need strong consistency).
4. [[Eventual Consistency]]: Updates will propagate eventually.


