
The search query of "Find every driver with 2km" for an Uber-like application.

For location data, we have [[Longitude|Latitude]] and [[Longitude]], and we care about is the straight-line distance between two points.
![[Pasted image 20260606161218.png|400]]
![[Pasted image 20260606161224.png|400]]

It's not appropriate to use two [[B-Tree]] indexes for this:
- One on lat gives us a horizontal strip of earth
- One on long gives us a vertical strip of earth

We have to load each of these strips into memory and have to check each of these against our point of interest by hand.

Another problem: 1D sort order doesn't preserve 2D closeness.
Take two cabs parked 5 blocks away in midtown manhattan.

![[Pasted image 20260606161400.png]]
If your 1D index sorts of longitude alone, every other cab in the city from Bronx to Staten Island gets crammed into the index between these two neighbors.
- ==Any naive flattening of 2D into 1D rips apart the relationship that we actually care about.==


Every modern system ([[PostGIS]], [[ElasticSearch]], [[Redis]]) have converted on one of two approaches to solve this problem:
1. Build a custom tree like a B-Tree that's specially tuned for location
2. Turn our lat/lon into a single key that a plain B-Tree can already sort.

Both approaches end up at the same prize: A balanced tree of pages on disk that you can range scan.


In 1974: The [[QuadTree]].
![[Pasted image 20260606161643.png]]
Take map and split into 4 quadrants. 
- If any of the quadrants have too many points in it, split that one into four again.
- Keep doing this recursively until each node has a management number of points.

On read
- To find points near a location, descend the tree one level at a time
- At each node, compare the query points lat/lon against the cell's midpoint, (N/S? E/W?) to determine which 
- At the leaf node, you check the neighboring leafs too to catch things sitting on the boundary.

It's nice because it adapts to density. Manhattan gets a deep tree, while a lake upstate stays as a single coarse cell.
- ==This has a bad downside on latency at scale==: Because the splits always happen at the geometric midpoint of a cell (not where the data actually sits), a dense cel like Manhattan keeps getting sliced over and over again until those points finally separate; you burn 10s of levels of depth before you actually reach a leaf node, whereas in Vermont, you have a depth of just 2.
So your query performance depends on the region you're looking in... and ==the hot regions (e.g. manhattan) are the slowest.==


You sometime still see QuadTrees today... but for most databases, we use something else.


In 1975: The [[KD-Tree]] (same author as Quadtree, one year later)

Idea: Instead of splitting all dimensions at once into four quadrants, alternate.
- At level 0, split on X
- At level 1: split on Y
- At level 2L split on X
- ...
![[Pasted image 20260606162159.png]]

This gives a binary tree for multidimensional data that's balanced if you split at the median. It was the workhorse for [[Nearest Neighbor]]-type searches for decades.


A modern improvement is a [[BKD Tree]], where instead of one point per node, points get packed into blocks sized to a disk [[Page]] and the whole tree is build once from a batch of data. [[ElasticSearch]] uses this today.



In 1984: The [[R-Tree]], the first spatial index designed for a database.
- Every index before this assumed that you data was a `POINT`, but real geospatial data has `LINES`, `POLYGONS`, etc.
- The disk issue: Gutman wanted the spatial analogue of a [[B-Tree]]:
	- Balanced
	- Fast to read off disk
	- Fast to update when uber drivers move
- His answer was to use ==Minimum-bounding Rectangles==
- Every single object gets wrapped in the smallest rectangle that contains it, with sides paralllel to the longitude-latitude axes on the map.
![[Pasted image 20260606162512.png]]

Then, nearby rectangles get grouped into larger, enclosing rectangles, which group into bigger ones, all the way up to the root.
![[Pasted image 20260606162539.png]]


The clever part is that this tree ==stays balanced like a B-Tree==
- Every Leaf lives at the same depth, so every query touches the same, predictable number of pages.
- Every node fits into a disk [[Page]]
- Inserts and deletions rebalance by splitting/merging pages, just like a B-Tree does.
- ![[Pasted image 20260606162656.png]]

One tradeoff: Unlike QuadTree styles, R-Tree's rectangles ==can overlap==, so if your queried point lands in two bounding rectangles at once, you have to descend both branches and check both subtrees, which can tank performance.

R-Trees are the workhorse of spatial data processing (e.g.  used in [[PostGIS]])



1985: [[R*-Tree]]
- Uses a smarter insertion algorithm to minimize those overlaps that we were talking about in R-Tree.
- This is what almost all modern R-Tree implementations are built on today.


____

Everything we've looked at so far... they're all custom tree structures. They work, but they're complicated and your database needs special indexing/query code to work with them.


What if we could skip them and turn lat/lon into a ==single number== such that numerically close integers meant gegographically close locations!
- If we had this, we could use our battle-hardened [[B-Tree]] implementations.


2008: [[Geohash]]
- Divide the world into a grid of 32 cells, and label each one with a character
![[Pasted image 20260606163057.png]]
- Now take the cell that you're in and divide into 32 smaller cells.
![[Pasted image 20260606163117.png]]
- Keep recursing as deep as you want; each character keeps recurring into the map.
	- 5 Characters: about 5km across
	- 9 Characters: 5 meters across

the final string `dr5ru` is your 2D location encoded as a 1D key.
- The payoff: ==Strings sharing a prefix are usually near eachother,== so you can use queries like `SELECT * FROM table WHERE geohash LIKE 'dr5ru%'`. This is just a B-Tree prefix scan, and is totally efficient!

Each character picks 1 of 32 options, which is exactly 5 bits!
- So a `dr5ru` is 25 of these bits grouped into 5 of these letters or readability.
- Read the same 25 bits as a number, and you have an integer! Same bits, same sort order.

This is why Redis can store a Geohash of integers, and Redis can store as Text.


There a ==catch== worth noting: Geohashes have edge cases at cell boundaries:
![[Pasted image 20260606163514.png]]
Picture a rider standing on the edge of a cell, starting at a Waymo 10 feet away. A naiev prefix scan on dr5ru would miss them entirely!

The standard fix is the ==3 by 3 Trick==
- Compute the cell your query point lands in, and then walk to the neighboring 8 cells, and query all 9 together as a unit.
![[Pasted image 20260606163554.png]]
- Now you can't miss anyone in a cell's distance.
- You can then ==post-filter== the results to drop the elements that are technically in your query window but actually too far away.



So Geohashes work great when mapping out a city, but start to hurt when you're looking at a globe.
- Geohash treats lat/lon as if it lived on a flat rectangle, but the earth isn't flat! A degree of longitude is ~0 at the poles and 111km at the equator, so a cell that's one-geohash character wide is a fat square the equator, and tiny at poles.

You might want cells of roughly equal are on earth.


2011: Google builds [[Sentinel|S2]] to fix this:

==Trick: Wrap the sphere in a cube, and then project the earth onto its six flat surfaces. On each face, subdivide cleanly into cells that stay roughly the same size everywhere on the globe.==

![[Pasted image 20260606165443.png]]
We have a 64-bit integer called an S2 Cell ID
- Like a geohash, these are hierarchical and truncatable: Truncate the id and get a coarser parent cell.




2008: Uber opensources [[H3]], which uses hexagons instead of squares.



What's wrong with a square?
![[Pasted image 20260606165609.png]]
A square shares 4 neighbors on edges, and 4 neighbors on corners, but those neighbors in the corners are further away!
- This asymmetry makes heat maps and "everyone within N cells of me" queries messy

![[Pasted image 20260606165705.png]]
Hexagons have 6 neighbors that are all at the same distance from eachother, which is good for some of the analytical queries that Uber likes to run


![[Pasted image 20260606165726.png]]
H3 tiles the entire globe in hexagons, and each hexagon subdivides somewhat cleanly into 7 hexagons, so you can zoom in as far as you need, with 16 resolutions in total, from continent-sized down to about a square meter.

We use 64-bit integer ID, and these IDs are hierarchical and truncatable. 

We sort their IDs:
- H3 walks the hexagon grid in a deterministic order using a [[Space-filling Curve]] and assigns each cell along that walk an integer that gets long as you move along the path.

The way that this works in production:
- Every Uber drive reports their location
- System converts that lat/lon into an H3 cell
- When a rider opens the app, we get their lat/lon and get their cell
- Expand to the 6 cells around them, and look for drivers.
- If needed, expand again...



![[Pasted image 20260606170039.png]]

![[Pasted image 20260606170127.png]]


Rule of thumb:
- If you need SHAPES, reach for a custom tree like [[R-Tree]]
- If you only have POINTS, reach for these [[Discrete Global Grid System|DGGS]] options like [[Geohash]], [[Sentinel|S2]], [[H3]]




































