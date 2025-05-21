A [[Geospatial Index]], similar to [[Geohash]]es and [[QuadTree]]s.

Instead of splitting space into fixed quadrants like in QuadTrees, R-Trees work with flexible, overlapping rectangles.
![[Pasted image 20250520184458.png]]
- ==Quadtrees adapt their rectangles to the actual data==, rather than rigidly dividing each region into four equal parts regardless of data distribution.
- When searching for nearby restaurants in SF, ==an R-Tree might first identify the large rectangle containing the city, then drill down through progressively smaller, overlapping rectangles until reaching individual restaurant locations==.

This flexibility offers a crucial advantage:
- R-Trees can efficiently handle both ==points== and larger ==shapes== in the same index structure.
- A single R-Tree can index everything from ==individual restaurant locations== to ==delivery zone polygons== and ==road networks==! The rectangles simply adjust their size to bound whatever shapes their contain.

The trade-off for this flexibility is that the ==overlapping rectangles sometimes force us to search multiple branches of the tree. Modern R-Tree implementations use sophisticated algorithms to balance this overlap against tree depth, optimizing for how databases actually read data from disk.==
- This balance of flexibility and disk efficiency is why R-Trees have become the standard choice for production spatial indices.