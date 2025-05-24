---
aliases:
  - Shard
---

When we have a database with too much write load, or if it's simply storing too much data, we need to split it across multiple computers/servers/hosts.

We want to be smart about how we split our data, as we want to ensure that the majority of our read queries go to a single node.

We also want to ensure that none of the nodes are responsible for handling a disproportionately large amount of the load. e.g. a [[Hot Spot]]


So how should we shard our database?
- **Key Range Sharding**
	- ![[Pasted image 20250523210717.png]]
	- Nice for when we want to still be able to support range queries; data with similar keys will still live on the same node.
	- But we're very prone to [[Hot Spot]]s, especially for names; there are many more names that start with A than start with Z, so you have to be careful!
- **Range of Key Hash**
	- For a given key, take the hash of a key, and then shard based on the hash of it.
	- A hash function should, in theory, evenly-distribute a range of keys.
	- Downside is that if you want to query all of the names that start with A, those hashes are all going to be different (resulting in different partition assignments), so we can't easily run a range query, since we'll have to go and access a bunch of different database shards in a [[Scatter-Gather]] query!
	- We **don't** want to take a hash of a key, modulo by the number of servers, and then pick that partition, because if we add a new server to the cluster, we'll have to shuffle a large amount of our data! Instead, we use [[Consistent Hashing]], where we organize a hash range into a conceptual "ring," and when we have an incoming key, we hash it onto the ring, and walk clockwise around the ring until you find a shard's location on the ring. 
		- This minimizes the amount of data rebalancing need when we either add or remove a machine from the cluster.
		- It's common to use vNodes as well.
			- Without vNodes, we might have uneven data distribution or hot spots, and when one node fails, all of its data goes to just one other node.
			- With vNodes (100-256 vNodes per physical node), each physical node is responsible for many small, scattered segments of the hash space. When you remove a node, its data gets distributed across ~ALL other nodes in the cluster, not just one.
			- Contrasting with hash+modulus.... with H+M, adding/removing one node reqiures rehashing/moving ~50% of all data, which is chaotic and affects the entire cluster. With vNodes, removing one node only affects the nodes data (1/N of total data), with the redistribution spread evenly across remaining nodes. So while all nodes might be affected in terms of who receives the new data, at least we aren't moving data around that doesn't need to move.


**Local Secondary Indexes**
- Same thing as adding an index per shard... rather than having an entire view of all of the rows in teh daabase, your local secondary index only applies to the rows on that particular shard. This doesn't slow down writes (which go to one shard), but if you want to read on that key... maybe our primary index is userId but secondary index is on the age of every user. If we want to find the age of all users < 24, we have to read from every shard and use their local secondary index.

**Global Secondary Indexes**
- These index the entire global table, and then they themselves arepartitioned according to that key, rather than the primary key... this amkes our reads much faster, because... we have evrything go to a particular shard.... we aren't making a scatter-gather.