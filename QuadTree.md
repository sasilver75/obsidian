A [[Geospatial Index]]

Here's how it works: Start with one square covering your entire area. When a square contains more than some threshold of  K points (typically 4-8), split it into four equal quadrants. Continue this recursive subdivision until reaching a maximum depth or achieving the desired point density per node:

For proximity searches, navigate down the tree to find the target quadrant, check neighboring quadrants at the same level, and adjust the search radius by moving up or down tree levels as needed.

The key advantage of quadtrees is their adaptive resolutoin: dense areas get subdivided more finely while sparse regions maintain larger quadrants.
- But while Geohashes leverage existing B-Tree implementations, Quadtrees require specialized tree structures. The implementation complexity explains why most modern DBs prefer Geohashes or R-Trees.

![[Pasted image 20250520184220.png]]



Not common in production nowadays (rather, people would use either [[Geohash]] or the conceptually similar [[R-Tree]]), Quadtrees had a significant impact on modern spatial indexing by introducing this recursive spatial subdivision strategy that forms the foundation for R-Trees, which optimize these ideas for disk-based storage and better handling of overlapping regions.