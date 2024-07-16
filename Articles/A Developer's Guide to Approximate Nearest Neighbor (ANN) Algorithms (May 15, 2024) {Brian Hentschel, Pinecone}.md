Link: https://www.pinecone.io/learn/a-developers-guide-to-ann-algorithms/

----

A crucial part of vector databases (emerging as a core part of the AI stack) are the [[Approximate Nearest Neighbor Search|ANN]] algorithms used to index data.
- These algorithms affect query performance, including:
	- Recall
	- Latency
	- Memory Usage/System Cost

All vector indexing algorithms have two components:
- The algorithm for updating data (inserts, deletes, updates)
- The algorithm for searching (reads)

An overview of storage media:
- Storage media are viewed as having a hierarchy, where items high in the hierarchy cost more, but have the best performance, and lower items have worse performance but lower cost. Depending on the type of storage you're doing (your access patterns, requirements) and your constraints, you might want to use various parts of the hierarchy for various purposes.

![[Pasted image 20240613184701.png|350]]
- Memory (DRAM): High bandwidth, low latency, high cost.
- Flash disk (SSD): Medium latency, medium bandwidth, medium cost.
- Object Storage: High latency, medium bandwidth, low cost.


Main types of vector indexing algorithms:
- Spatial Partitioning (cluster-based indexing, clustering indexes)
- Graph-based Indexing (eg [[Hierarchical Navigable Small Worlds|HNSW]])
- Hash-based Indexing (eg [[Locality Sensitive Hashing|LSH]])

We'll skip hash-based indexing because its performance on all aspects (reads, writes, storage) is currently worse than that of both graph-based and spatial partioning-based indexing.
- ==Almost no vector databases us hash-based indexing nowadays.==

## Spatial Partitioning
- Organizes the data into regions; vector are stored alongside other nearby vectors in the same region of space. 
- Each partition is represented by a representative point, usually the *centroid* of the data points stored in the partition.
- Queries operate in two stages:
	- Find the representative points closest to them
	- Scan the partition for relevant related points

Several benefits
- They have very little space overhead; the structures created for updating and querying data just consist of representative points (eg centroids).
- Work well with object storage, as this class of methods usually results in making reads that are both smaller in number and longer than graph indexes.
	- However, generally have lower query throughput than graph-based indexes ((How is this possible?))

## Graph-based Indexing
- Graph-based indexes organize data as a graph. Reads start at an entry point and work by traversing the graph greedily by performing a best-first search.
	- Algo keeps a priority queue of bounded size `m`, containing promising points whose neighbors haven't been viewed yet.
	- At each step, the algo takes the unexpanded point closest to the query, and looks at its neighbors, adding any points closer than the current m'th closest point into the priority queue.
	- Search stops when the queue contains only points whose neighbors have been expanded.
- Graph algorithms have been shown to have the ==best algorithmic complexity in terms of computation, in that they compute the distance to the fewest number of points to achieve a certain recall.==
	- ==As a result, they tend to be the fastest algorithms for in-memory vector search, and can engineered to work well on SSDs as well.==
- A major difference is that graph-based algorithms are sequential; it makes hops along the graph, and each hop is a small read -- so graphs don't do well with storage mediums with high latency for read requests, like object stores.
- Additionally, each data point in a graph holds multiple edges, so they take more space to store than partitioning-based indices, and inserts/deletes touch multiple nodes in a graph, leading to higher costs for updates, especially for storage mediums with high latency.

## Mixed-Index Types
- Indexes can be a mix of types too! 
![[Pasted image 20240613191227.png|300]]
Above: This is one of the most common mixed indexes. Here, vectors (red crosses) are organized into spatial partitions (shown by black boundaries, with black circle centroids). The representative points (centroids) are themselves indexed; since there are few of these, it's easier to pay the price to store them in a more expensive storage media like memory, and to store neighbors for each point. So we commonly index the representative points using a graph (eg [[Hierarchical Navigable Small Worlds|HNSW]]), with the vectors themselves indexed using something like [[Inverted File Index|IVF]].
- Indexes like ==SPANN== are like this, with a graph on the centroids but uses spatial partitioning for the bae points -- we would call this as a partitioning-based indexing strategy rather than a graph one, because the base pointers are indexed by partition.

## Hierarchical Navigable Small Worlds (HNSW)
- One of the most popular algorithms is [[Hierarchical Navigable Small Worlds|HNSW]]; it's the main indexing algorithm that most vector search solutions use.
- HNSW offers great recall at high throughput, but has large memory costs and high update costs that don't work well with production applications with highly-dynamic data.

## Inverted File Index (IVF)
- [[Inverted File Index|IVF]] is easy to use, has high search quality, and has reasonable search speed. It's a popular index and has little space overhead; Can work well with object stores, but has lower query throughput compared to graph-based indexing.

## DiskANN
- DiskANN is a graph-based index that can be designed for entirely in-memory use cases or for data on SSD use cases. It's a leading graph algorithm and shares benefits and drawbacks that most graphs possess.

## SPANN
- SPANN is a spatial partitioning based index meant for use with data on SSD; builds a graph index over the set of representative points (centroids) that are used for partitioning.
- It's good for disk-based nearest-neighbor searches, being faster than DiskANN at lower recalls, and slower at higher recalls.