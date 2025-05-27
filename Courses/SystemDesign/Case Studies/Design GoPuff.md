SDIAH Article: https://www.hellointerview.com/learn/system-design/problem-breakdowns/gopuff
(No Video)

I thought this one was pretty poor quality, TBH. Didn't like the service granularity/lack of referential integrity/sharing databases.

(Something interesting about this one is that while we didn't choose to use Geohashes for proximity queries, because we only had 10k DCs, and we could easily keep them in-memory and do the computation in a few milliseconds.)
(It wasn't clear to me again if we wanted to have a singular InventoryItem for every physical bag, or instead if we wanted to have one for each DC, with a quantity.)
(Something else to note is that this doesn't really talk about geographic partitioning.)

--------

Recall our framework:
- **==Functional Requirements==** (FRs): "A user can..." the core functionalities of the system.
	- Also nice to establish explicitly what is ==OUT== of scope
- **==Non-Functional Requirements==** (NFRs): The qualities of the system and additional requirements
- ((Optional: Back of the Envelope Calculations; Prefer to do it as-needed in HLD or DD))
- **==Core Entities==**: The tables/core entities that our system handles (no attributes yet)
- **==API Design==**: Interface between clients and our systems (One+ for each FR)
- ((Optional: Data Flow))
- **==High-Level Design==**: Boxes and arrows; Satisfy our FRs by going through each of the API routes and making sure we can service them.
- **==Deep Dive==**: Go through 1-3 of the NFRs and make sure that we can get those qualities we want.

-------


# Core Requirements
- GoPuff is basically DoorDash, but where we usually call "restaurants" as "distribution centers", since some of them might basically just be a closet with a bunch of snacks in them. Your order may have items from multiple DCs, somewhat transparently to you.
- Requirements:
	- Customers should be able to ==query availability of items, deliverable in one hour, by location== (i.e. the effective availability is the union of all inventory nearby distribution centers (DCs))
	- Customers should be able to ==order multiple items== at the same time.
- Out of Scope
	- Handling payments/purchases
	- Handling driver routing and deliveries
	- Search functionality and catalog APIs (this system is strictly concerned with availability and ordering)
	- Cancellations and returns

==For this problem, the emphasis is on aggregating availability of items across local distribution centers, and allowing users to place orders without double-booking items.== In **other** problems, you might be more concerned with the product catalog, search functionality, etc.


# Non-Functional Requirements
NFRs
- ==Availability requests should be fast== (<100ms) to support use-cases like search.
- Ordering should have [[Strong Consistency]]: Two customers should not be able to purchase the same physical product.
- System should be able to support ==10k distribution centers== and ==100k items== in the catalog across DCs.
- Order volume will be ==O(10M orders/day)==

OUT OF SCOPE
- Privacy and security
- Disaster recovery

![[Pasted image 20250525100749.png]]

((What's not clear to me is if it's a product requirement that an order added to a cart is reserved. Because if an order added to a cart isn't reserved, it feels like we don't even need carts, we can just POST a CreateOrder with a bunch of Inventory ids, and that might fail if those Inventory items aren't available.))

# Core Entities
- ==DistributionCenter==: A location that has Inventorys
- ==Item==: A type of item, e.g. "Cheetos"; these are what our customer cares about.
- ==InventoryItem==: A physical instance of an Item, located at a DistributionCenter.
- ==Order==: A customer's Order, which has various OrderItems (and shipping/billing information?)
- ==OrderItem==: ?

![[Pasted image 20250525102252.png|400]]



==Important note==: The distinction between an Item and Inventory:
- While our customers are strictly concerned with items that they might see in a Catalog (e.g. Cheetos), we need to keep track of where the PHYSICAL items are actually located. Our InventoryItem entity is a physical item at a specific location.

==Tip:== Start from the most concrete physical or business entities (e.g. items, users) and work your way up to more abstract entities (e.g. orders, carts, etc.) to ensure you don't miss important entities.

# Defining the API
- These should closely track our FRs, which are:
	- Customers can query the availability of items, deliverable in one hour, by location
	- Customers can order multiple items at the same time

So it seems like we need two APIs:
- The first lets us get the available items given a location (and maybe keyword, if we're doing search?)
- The second allows us to place an order

```

%% Ability to get available Items near my LatLon, optionally with keyword, using pagination %%

GET /v1/availability?lat=LAT&lon=LON&keyword={}&page_size={}&page_num={} -> Item[]

Item: {name: NAME, quantity: QTY}


%% Ability to create an order %%
%% ((I guess the assumption is that two Cheetos would just be two Items?)) %%
%% Note that we're repassing our location, because we'll need to confirm that we have enough Inventory nearby the user's location (deliverable in an hr) to satisfy the user's order) %%

POST /v1/order -> Order | Failure
{
	lat: lat
	lon: lon
	items: Item[]
	...
}

```


# High-Level Design
- We do this by going through our API routes and making sure that we can satisfy each of them in turn.


**==First==: GET /v1/availability?lat=LAT&lon=LON&keyword={}&page_size={}&page_num={} -> Item[]** 

Tod do this, we have two steps:
- We need to find the DCs that are close enough to deliver in 1 hour.
- We need to find the Inventory that lives in those DCs
- (Optionally, later, we need search functinoality)

To find nearby DCs, we need an internal API which takes a LAT and LONG and returns a list of DCs within one hour.
- Let's assume that we have a table of DCs with their lat/long.
- Crudely, we can measure the distance to the user's input with some simple math.
- A very basic version might use a Euclidean distance, while a more complicated one might use a [[Haversine Distance]] distance.
- Taking a simple threshold on this query would give us DCs within X distance as the crow flies.
- This doesn't quite satisfy our functional requirement, but we'll come back to this in our deep dive.


![[Pasted image 20250525105342.png]]


Next, we need to check the inventory of the DCs that we just found.
- We can do this by querying our Inventory table and Items table.
- Let's assume we're using a Postgres database, allowing us to join our inventory table with our item table WHERE the dc.id are in the ids we found before.

![[Pasted image 20250525105944.png]]
((Sam: Okay, so you have a DCId in this database, but it's foreign-keying to another database, which is... bad practice, right? You can't really enforce that foreign key for your data referential integrity, and we can't do joins across it.))


==**NOTE:** In many ecommerce systems, the "**Catalog**" is stored separately from the **inventory**==, because of the different consumers and workloads. We'll store them in the same database here to make our job easier and to adhere to our requirements, but we might note to the interviewer that we'd ideally separate these, add a search index (e.g. Elasticsearch) to allow searching the catalogue.


![[Pasted image 20250525110132.png]]
((This is fucking stupid.))

When a user makes a request to get availability for items A, B, and C from latitude X and longitude Y, here's what happens:
1. We make a request to the Availability Service with teh user's location X and Y and any relevant filters
2. The availability service fires a request to the Nearby Service with the user's location X and Y
3. The nearby service returns us a list of DCs that can deliver to our location
4. With the DCs available, the availability service queries our database with those DC Ids to get the available Inventory for those DCs.
5. We sum the results and return them to our client.

Q: "Why not use a geohash of 4 characters for each DC (20km radius), and use the user's geohash to quickly determine what DCs can meet the user's need the best? Though since DCs are almost always static, storing them in memory is probably just fine?"
A: "Yeah with 10k DCs that very rarely change, storing them in memory works great and can be filtered in < 1 ms."


**==Next==**: POST /v1/order -> Order | Failure 
- ==Customer should be able to order items==

How do we enable placing orders?
- For this, we require strong consistency to make sure two users aren't ordering the same item.
- To do this, we need to ==check inventory==, ==record the order==, and ==update the inventory together atomically==.
- Though latency isn't a big concern here (users will tolerate more latency here than they will on the reads), we definitely want to make sure that we're not promising the same inventory to two users. How do we do this?

The idea of ensureing that we're not **double booking** is a common one in system design.
- To ensure we don't allow two users to order the same inventory, we need some form of locking!
- The idea being that we need the LOCK the inventory while we're checking it, and record the order in such a way taht only one user can hold the lock at at time.

- =="Good" Solution: Two data stores with a distributed lock==
	- We can have separate databases for orders and inventory. 
	- When an order is placed, we lock the relevant inventory records, create the order record, decrement the inventory, and release the lock.
	- This is a good solution because it lets us use the best data store for each use case
		- e.g. a KV Store for inventory and a relational database for orders
	- CHALLENGES
		- Nasty failure modes! 
			- What if our service crashes after we created the order but before we decremented the inventory!
				- A subsequent user might order the inventory we had promised to the first user.
				- We'll need to sweep for these failures and reverse them.
			- What if two orders have overlapping inventory requirements?
				- We might deadlock if both User1 and User2 are trying to buy A and B, but User1 has the lock for A and User2 has the lock for B -- neither can proceed!
- ==**GREAT** Solution: Singular Postgres transaction==
	- By putting both orders and inventory in the same database, we can take advantage of the [[ACID]] properties of our [[PostgreSQL|Postgres]] database.
	- Using a singular transaction with [[Serializability|Serializable]] isolation level, we can ensure that the entire transaction is atomic.
	- This means that if two users try to buy the same item at the same time, one of them will be rejected.
	- This is because the transaction will fail to commit if the inventory is not available!
	- ![[Pasted image 20250525111223.png|500]]
	- CHALLENGES
		- While consolidating our data down to a single database has benefits, it's not without drawbacks; we're partly coupling the scaling of inventory and orders, and we can't take advantage of the best data store for each use case.
	- ==**TIP**==: When [[Atomicity]] of transactions is a requirement, it's *helpful* to have your data colocated in an [[ACID]] datastore that can provide those guarantees. While it's *possible* to manage [[Distributed Transaction]]s across multiple data stores (e.g with [[Two-Phase Commit|2PC]], the additional complexity and overhead to support it is not what we want to focus on during this 

By choosing the "great" option and leaning in to our existing Postgres database, we can keep our system simple and still meet our requirements.

==For an order, the process lookings like this:==
- The user makes a request to the **Orders Service** to place an order for items A,B,C
- The **Orders Service** makes a singular transaction where we submit to our Postgres leader:
	- Check the inventory for items A,B, and C > 0 (or above whatever requirement in the order)
	- If any of the items are out of stock (no **inventory**), the transaction fails.
	- If all items are in stock, the transaction updates the status for available **inventory** items (A,B,C) to "ordered"
		- ((So each **Inventory** item is literally a bag of chips, okay.))
	- A new row is created in the **Order** table and **OrderItems** table recording the order for A,B,C
	- The transaction is committed.

((I wonder if it makes sense to have a singular **Inventory** record per bag of chips, with a status, and well as a singular **OrderItem** record once it's been ordered... as opposed to just having a nullable **OrderId** column on the **Inventory** table? It depends how much additional information we're going to be adding to **OrderItem** perhaps, which at this point doesn't seem like much?))

==If any of the items become unavailable in the user's order, the entire order fails.==
- This is a thing that we should bring up with the interviewer, and ask whether it's an appropriate outcome, product-wise.

![[Pasted image 20250525112020.png]]
- Above:
	- ((I gripe about having a separate DB for our DC and for our inventory, which breaks referential integrity))
	- ((I have a slight gripe about our OrdersService and AvailabilityService both using the same database. This is an antipattern in MS architectures.))
	- ((I would probably just have  single primary service))

We have three services, one for Availability requests, one for Orders, and a shared service for Nearby DCs. Both our Availability and Orders service use the Nearby service to look up DCs that are close enough to the user. We have a singular Postgres database for inventory and orders, [[Partition]]ed by region. Our Availability service reads via [[Replication|Read Replica]]s, our Orders service writes to the leader using **atomic transactions** to avoid double writes. A great foundation!
- ((Uhh we haven't talked about partitioning or replication yet for our databases. That should not be in the high-level design.))

# Deep Dives
- Now we want to satisfy our Non-Functional Requirements:
	- ==Availability requests should be fast== (<100ms) to support use-cases like search.
	- Ordering should have [[Strong Consistency]]: Two customers should not be able to purchase the same physical product.
	- System should be able to support ==10k distribution centers== and ==100k items== in the catalog across DCs.
	- Order volume will be ==O(10M orders/day)==


### Deep Dive: Make availability lookup incorporate traffic and drive time
- So far our system only determines DCs based on a simple distance calculation, but this may be different from the drive time because of **geography** (e.g. rivers, borders). Also, **Traffic** might influence the travel times. Since our functional requirements mandate an hour of drive time, we need something more sophisticated.
- =="Bad" Solution: Simple SQL Distance==
	- We put all the lat/long of our DCs into our DC table, and measure the distance to the user's input with some simple Math (e.g. either Euclidean or [[Haversine Distance]]).
	- Taking a simple threshold this query would give us DCs within X as the Crow Flies.
	- **Why it's not optimal**: This doesn't take into account traffic, road conditions, etc., not does it take into account the fact that we might have multiple DCs in the same city.
- =="Bad" Solution: Use a Travel Time Estimation Service against **ALL** DCs==
	- Since our DCs are rarely going to change (they're buildings!) we can sync periodically (every 5 minutes) from a DC table to the memory of our **Nearby Service**. We can then use a **Travel Time Estimation Service** to find the travel times from our input location to each of our DCs by iterating over all of them.
	- ![[Pasted image 20250525114034.png|500]]
	- **Challenges:**
		- ((I have no idea what the fuck this strategy is even saying. We have the DCs in-memoryin nearby service, so... Okay? wtf?))
		- We're making far too many queries to the travel time estimation service. Most of the DCs we're querying aren't close enough to plausibly be delivered in an hour.
- **=="Great" Solution**: Use a Travel Time Estimation Service against **NEARBY DCs**== 
	- We build on the previous solution to sync periodically (like every 5 minutes) from a DC table to the memory of our service.
	- When an input comes in, we can prune down the "candidate" DCs by taking a fixed radius (say 60 miles, the most optimistic distance we could drive in over an hour), and then limit ourselves to only evaluating those. 
	- We can take these restricted candidates and pass them to the external travel time service to create our final estimate.
		- ((Ah, I see what they were doing in the past one))
		- ((So this is the same as the previous one but we're doing some rough in-memory filtering of the data so that we don't have to pass all of it to the TTES.))

### Deep Dive: Make availability lookups fast and scalable
- Currently, once we have nearby DCs, we use those DCs to look up availability directly from our database.
- This introduce a lot of load into our databases!
- ==We might want to do some quantitative estimation here, once we'd spotted a potential bottleneck... This gives you and your interviewer a common set of data from which to weigh tradeoffs and its shows you're able to make reasonable assumptions about the system.==

To find our how many queries for availability we might have, we'll back in from our orders/day requirement:
- 10M orders a day

All in all, we might estimate that each user will look at ==10 pages across search== before purchasing an item. Maybe only 5% of those users will end up buying, where the rest are just shopping.
- Queries: `10M orders/day` / `100k seconds/day` * `10` / `0.05` = `20k queries/second`

This is a pretty sizeable number of queries per second... given that this is assuming a uniform distribution of queries, which likely isn't even realistic.

- **==Great Solution==**: Query the currently available inventory through a cache
	- We can add a Redis instance to our setup, where the availability service can query the cache for a given set of inputs, and, if the cache hits, return the result. If the cache misses, we'll do a lookup on the underlying datbase and then write the results into the cache.
	- Setting a **Time to Live** of ~ 1 minute ensures these results are fresh.
	- ![[Pasted image 20250525121106.png|450]]
	- ((Wait, why would this work? What would the Cache key be? Oh, so the availability service gets the nearby DCs given the client-provided latlon, then for a list of DCs, gets the items in the cache? So each DC has its own entry in Redis, or something? As long as each DC has been queried in the last minute, there should be an entry in there, so that's fine and probably realistic, given that users are just fetching))
	- ((We would need to decide, as usual, our [[Cache Write Strategy]] ([[Write-Around Cache]]), deciding what to do on a cache miss ([[Cache Aside]]), a cache invalidation strategy, and a cache eviction strategy ([[Time to Live]])))
- **==Great Solution==**: Postgres Read Replicas and Partitioning
	- Instead of caching, we can just increase the write and read throughput of our database by [[Partition]]ing (by DCid) and [[Replication]] for [[Replication|Read Replica]].


-----------------------


## [What is Expected at Each Level?](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

Your interviewer may even have you go deeper on specific sections, or ask follow-up questions. What might you expect in an actual assessment?

### Mid-Level

**Breadth vs. Depth**: A mid-level candidate will be mostly focused on breadth. As an approximation, you’ll show 80% breadth and 20% depth in your knowledge. You should be able to craft a high-level design that meets the functional requirements you've defined, but the optimality of your solution will be icing on top rather than the focus. **Probing the Basics**: Your interviewer spend some time probing the basics to confirm that you know what each component in your system does. For example, if you use DynamoDB, expect to be asked about the indexes available to you. Your interviewer will not be taking anything for granted with respect to your knowledge. **Mixture of Driving and Taking the Backseat**: You should drive the early stages of the interview in particular, but your interviewer won't necessarily expect that you are able to proactively recognize problems in your design with high precision. Because of this, it’s reasonable that they take over and drive the later stages of the interview while probing your design. **The Bar for GoPuff**: For this question, interviewers expect a mid-level candidate to have clearly defined the API endpoints and data model, and created both routes: availability and orders. In instances where the candidate uses a “Bad” solution, the interviewer will expect a good discussion but not that the candidate immediately jumps to a great (or sometimes even good) solution.

### Senior

**Depth of Expertise**: As a senior candidate, your interviewer expects a shift towards more in-depth knowledge — about 60% breadth and 40% depth. This means you should be able to go into technical details in areas where you have hands-on experience. **Advanced System Design**: You should be familiar with advanced system design principles. Certain aspects of this problem should jump out to experienced engineers (read volume, trivial partitioning) and your interviewer will be expecting you to have reasonable solutions. **Articulating Architectural Decisions**: Your interviewer will want you to clearly articulate the pros and cons of different architectural choices, especially how they impact scalability, performance, and maintainability. You should be able to justify your decisions and explain the trade-offs involved in your design choices. **Problem-Solving and Proactivity**: You should demonstrate strong problem-solving skills and a proactive approach. This includes anticipating potential challenges in your designs and suggesting improvements. You need to be adept at identifying and addressing bottlenecks, optimizing performance, and ensuring system reliability. **The Bar for GoPuff**: For this question, a senior candidate is expected to speed through the initial high level design so we can spend time discussing, in detail, how to optimize the critical paths. Senior candidates would be expected to have optimized solutions for both the atomic transactions of the orders service as well as the scaling of the availability service.

### Staff+

**Emphasis on Depth**: As a staff+ candidate, the expectation is a deep dive into the nuances of system design — the interviewer is looking for about 40% breadth and 60% depth in your understanding. This level is all about demonstrating that "been there, done that" expertise. You should know which technologies to use, not just in theory but in practice, and be able to draw from your past experiences to explain how they’d be applied to solve specific problems effectively. Your interviewer knows you know the small stuff (REST API, data normalization, etc) so you can breeze through that at a high level so we have time to get into what is interesting. **High Degree of Proactivity**: At this level, your interviewer expects an exceptional degree of proactivity. You should be able to identify and solve issues independently, demonstrating a strong ability to recognize and address the core challenges in system design. This involves not just responding to problems as they arise but anticipating them and implementing preemptive solutions. **Practical Application of Technology**: You should be well-versed in the practical application of various technologies. Your experience should guide the conversation, showing a clear understanding of how different tools and systems can be configured in real-world scenarios to meet specific requirements. **Complex Problem-Solving and Decision-Making**: Your problem-solving skills should be top-notch. This means not only being able to tackle complex technical challenges but also making informed decisions that consider various factors such as scalability, performance, reliability, and maintenance. **Advanced System Design and Scalability**: Your approach to system design should be advanced, focusing on scalability and reliability, especially under high load conditions. This includes a thorough understanding of distributed systems, load balancing, caching strategies, and other advanced concepts necessary for building robust, scalable systems. **The Bar for GoPuff**: For a staff+ candidate, expectations are set high regarding depth and quality of solutions, particularly for the complex scenarios discussed earlier. Your interviewer will be looking for you to be diving deep into at least 2-3 key areas, showcasing not just proficiency but also innovative thinking and optimal solution-finding abilities. They should show unique insights for at least a couple follow-up questions of increasing difficulty. A crucial indicator of a staff+ candidate's caliber is the level of insight and knowledge they bring to the table. A good measure for this is if the interviewer comes away from the discussion having gained new understanding or perspectives.