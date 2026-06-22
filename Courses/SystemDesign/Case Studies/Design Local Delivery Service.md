e.g. Design GoPuff

> Gopuff delivers goods typically found in a convenience store via rapid delivery and 500+ micro distribution centers (DCs).


# Functional Requirements
1. Customers should be able to query the availability of items, deliverable in 1 hour, by location (i.e. the effective availability is the union of all inventory of nearby DCs)
2. Customers should be able to order multiple items at the same time.

Out of scope: Handling payments/purchases, handling driver routing and deliveries, search functionality and catalog APIs, cancellation and returns.

# Non-Functional Requirements
- Availability requests should be fast (<100ms) to support use-cases like search
- Ordering should be [[Strong Consistency|Strongly Consistent]]; two customers should not be able to purchase the same physical product.
- System should be able to support 10k DCs and 100k items in the catalog, across DCs
- Order volume will be order of ~10M orders/day

Out of scope: Privacy and security, disaster recovery


# Defining Core Entities
- Item: An item type (e.g. Cheetos), which are what our customers actually care about.
- Inventory: A specific physical item at a specific physical location. We can sum up Inventory to determine the quantity available to a specific user for a specific `Item`.
- Distribution Center: A physical location where items are stored. We'll use these to determine which items are available to a user. `Inventory` are stored in DCs.
- Order: A collection of `Inventory` which have been ordered by a user.
- Order Item: 


# API Contract
So for this, we typically want to look at our functional requirements.
- We need an API that lets us get the availability of items given a location (and maybe a keyword?)
	- Might want to include [[Pagination]].
- An API that lets us place an order.

To get the availability of items around us:
```json
GET /v1/availability?lat=LAT?long=LONG?keyword={}&page_size={}&page_num={}
->
{
	items: {
		name: NAME,
		quantity: QUANTITY
	}[]	
}
```

To create an order:
```json
POST /v1/order
{
	lat: LAT
	long: LONG
	items: ITEM1, ITEM2, ITEM3...
	...
}

-> Order | Failure
```
Note above that we're passing our lat/long to both APIs; before the order can be processed, we'll need to confirm the inventory is available close enough to the user's location to deliver within 1 hour.


# High-Level Design
- With this, we want the infrastructure to be able to satisfy our functional requirements.

FR 1: Customer should be able to query the availability of items
- To do this, we have two steps:
	1. We need to find DCs that are close enough to deliver in 1 hour.
	2. Once we have a list of DCs, we can check the inventory of  them and return the union of these inventories to the user.
- We'll eventually need each of these to be pretty quick, since we want this to run in <100ms
- To find nearby DCs, we can build a simple internal API which takes a LAT/LONG and returns a list of DCs within 1 hour, assuming we have a table of DCs with their lat/long. 
	- This might most crudely be done as euclidean distance, while a slightly more sophisticated case might use the [[Haversine Distance]]... but we'll come back to this in our deep dive.
![[Pasted image 20260621212159.png]]

Next, we need to check the inventory of the 1-hour-reachable DCs we found:
- We can do this by querying our Items and Inventory tables.
	- (Recall: Items are the abstract items and Inventory are the physical units of it in DCs)

![[Pasted image 20260621212543.png]]
Above: I think this is basically saying: For a given set of DCs, we want to retrieve the ITEMS that have related INVENTORIES in the DCs that we have in our collection. So we join our Inventory to Items and then filter to those having Inventory.DCId in {dcids}, and return just the unique item information.

So we have:
1. Availability Service: Handles requests from our users for availability, given a specific location.
2. Nearby Service: Synchronizes with the database of nearby DCs and uses an external "Travel Time Service" to calculate travel times from DCs (potentially including traffic)
3. Inventory Table: A replicated SQL database table which returns the inventory available for each item and DC

![[Pasted image 20260621215113.png]]
When a user makes a request to get availability for items A/B/C from latitude X and longitude Y, here's what happens:
- We make a request to the Availability Service with the user's location X and Y and any relevant filters
- The availability service fires a request to the Nearby Service with the user's location X and Y.
- The nearby service returns us a list of DCs that can deliver to our location
- With the DCs available, the availability service queries our database with those DC IDs
- We sum/union the results and return them to our client.

FR #2: Customers should be able to order items.
- The last thing we need to complete our requirements is for us to enable placing orders.
- For this, we require [[Strong Consistency]] to make sure that two users aren't ordering hte same item. To do this, we need to check inventory, record the order, and update the inventory together atomically.
	- Note that latency isn't as big of a concern here; users can tolerate a spinner.
- The idea is that we need to ensure that there isn't ==double booking,== by using some form of locking.
	- We "lock" the inventory while we're checking it and recording the order in such a way that only one user can hold the lock at a time.

You MIGHT BE TEMPTED to have two databases here (one for orders, one for inventory) and to have to manage a [[Distributed Transaction]], but if we instead choose to put them in the *same* database, we can take advantage of the [[ACID]] properties of a singular [[PostgreSQL|Postgres]] database, with isolation level `SERIALIZABLE`... If two users try to order the same item at the same time, one of them will be rejected. This is because the transaction will fail to commit if the inventory is not available.

![[Pasted image 20260621215751.png]]
((Typically it's an antipattern for multiple services to share the same database... but it's more fine if (e.g.) Orders service were never to look at `Inventory` tables, etc... where you have an implicit service boundary. The point is that each piece of business data should have one clear owner.... but wait, that's not what they're doing here, lmao.))

==NOTE==: When atomicity of transactions is a requirement, it's helpful to have your data colocated in an ACID database store. While it's possible to manage transactions across multiple data stores, the additional complexity/overhead to support it is often not needed.

So our process:
1. The user makes a request to `OrdersService` to place an order for items A, B, and C
2. The Orders Service creates a singular transaction, where:
	1. We check the inventory for Items A/B/C are > 0
	2. If any of the times are out of stock, the transaction fails.
	3. If all items are in stock, the transaction records the order and updates the status for inventory items A/B/C to "ordered".
	4. A new row is created in the Orders table recording the order for A/B/C
	5. The transaction is committed
3. If the transaction succeeds, we return the order to the user.

Downside: If any of the items become unavailable in the user's order, the entire order fails. This may or may not be what you want. We'll want to make sure to return a more meaningful error message in this case.

Now we can pull together our implementation:
![[Pasted image 20260621220821.png]]



Deep Dives
1. Make availability looks incorporate traffic and drive time
	- So far our availability lookups are based on a simple distance calculation... but if our DC is over a river or a border, it might be close in miles but not close in drive time.
	- We can introduce a Travel Time Estimation Service, and use it against Nearby DCs
		- So we first do what we did before, getting all of the candidate DCs within some fixed radius of X miles.
		- We'll take these restricted candidates and then pass to the external travel time service to create our final distance estimate, and then filter the resulting list accordingly.
![[Pasted image 20260621221713.png]]

2. Make availability looks fast and scalable. TO figure out how many queries for availability we might have, we can back into it from our orders/day requirement, which we set at 10M orders a day. We might estimate that a user will look at 10 pages before purchasing 1 item, and maybe only 5% of these users will end up buying, whereas the rest are shopping.

```
Queries: 10M orders/day / (100k seconds/day) * 10 / 0.05 = 20k queries/second
```
This is a pretty sizeable number of queries per second. 
What are our tools for scaling reads?
- Caching
- Read Replicas
- Indexing
- Denormalization
- ...

![[Pasted image 20260621222143.png]]
****
































