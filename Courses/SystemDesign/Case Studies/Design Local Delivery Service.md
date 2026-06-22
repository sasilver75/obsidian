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
1. Availability Service
2. Nearby Service
3. Inventory Table: A replicated SQL database table which returns the inventory avilable for each item and DC





























