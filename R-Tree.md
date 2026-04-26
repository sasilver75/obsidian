---
aliases:
  - Rectangle Tree
  - Hilbert R-Tree
---

A [[Geospatial Index]], similar to [[Geohash]]es and [[QuadTree]]s., Invented in 1984
- Variation: Hilbert R-Tree, as used in [[FlatGeobuf]] file-internal indexes.

- It indexes ==maximum bounding rectangles== (MBRs).
	- For a point, it's just the point
	- For a polygon like a neighborhood boundary, the MBR is the smallest rectangle that contains the whole polygon.
```
Actual polygon:           Bounding box:
    /\                    ┌──────┐
   /  \                   │  /\  │
  /    \                  │ /  \ │
 /______\                 │/____\│
                          └──────┘
```
- R-Trees then group nearby bounding boxes together into parent bounding boxes recursively, until the whole dataset is covered.
```
Level 3 (root):    [ Entire LA region ]
                   /                  \
Level 2:  [ West LA + South Bay ]  [ East LA + SGV ]
          /          \               /          \
Level 1: [Santa Monica] [Culver City] [DTLA] [Pasadena]
          |               |              |         |
Level 0: [points...]  [points...]   [points...] [points...]
```
- Now if we weant to "find all points within 500M of City Hall", we:
	- Convert the search radius to a bounding box
	- At the root, check which child bounding boxes overlap the search, and prune away branches that can't contain matches.
	- Recurse only into overlapping branches.
	- At the leaf level, do the exact geometry check, where we might check 200 candidates instead of 1.5 million.

Instead of splitting space into fixed quadrants like in QuadTrees, R-Trees work with flexible, overlapping rectangles.
![[Pasted image 20250520184458.png]]
- ==Quadtrees adapt their rectangles to the actual data==, rather than rigidly dividing each region into four equal parts regardless of data distribution.
- When searching for nearby restaurants in SF, ==an R-Tree might first identify the large rectangle containing the city, then drill down through progressively smaller, overlapping rectangles until reaching individual restaurant locations==.

This flexibility offers a crucial advantage:
- R-Trees can efficiently handle both ==points== and larger ==shapes== in the same index structure.
- A single R-Tree can index everything from ==individual restaurant locations== to ==delivery zone polygons== and ==road networks==! The rectangles simply adjust their size to bound whatever shapes their contain.

The trade-off for this flexibility is that the ==overlapping rectangles sometimes force us to search multiple branches of the tree. Modern R-Tree implementations use sophisticated algorithms to balance this overlap against tree depth, optimizing for how databases actually read data from disk.==
- This balance of flexibility and disk efficiency is why R-Trees have become the standard choice for production spatial indices.



![[Pasted image 20260425235117.png]]

![[Pasted image 20260425235121.png]]


