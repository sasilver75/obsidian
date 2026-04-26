A "K-Dimensional Tree", a [[Geospatial Index|Spatial Index]] that ==recursively partitions space by splitting along ONE axis at a time.==

The structure:
- At each node, pick a dimension and a split value; all points with a smaller value go left, and larger go right. 
- Alternate dimensions as you descend:
```
  Split on X at 5
  ├── Left (X < 5): split on Y at 3
  │   ├── Left (Y < 3): split on X at 2  → leaf
  │   └── Right (Y > 3): split on X at 4 → leaf
  └── Right (X > 5): split on Y at 7
      ├── ...
      └── ...
```

Each leaf contains a small bucket of points.

The classic use case for KD-Trees is [[K-Nearest Neighbors|kNN]] search:
- To find the k Nearest neighbors of a query point:
	- Traverse down to the leaf containing the query point
	- Check those points, record best distance found
	- Backtrack up the tree; at each node, check if the other branch's half-space could *possibly* contain a closer point (if splitting plane is closer than current best, it might!)
	- Prune branches that can't improve the answer (this is what makes it fast, skipipng huge chunks of the dataset).

[[Curse of Dimensionality]]: KD-trees degrade badly in high dimensionality, where the "could the other branch be closer?" check almost always says yes, and you end up visiting most of the tree anyways.
- Rule of thumb is that ==KD-Trees work well up to ~20 dimensions, and beyond that, approximate methods like [[Hierarchical Navigable Small Worlds]] win.==
	- This is why [[FAISS]] doesn't use [[KD-Tree]]s for embeddings search, since embeddings are commonly >= 128 dimensions.

![[Pasted image 20260426001114.png]]

## compared to [[Octree]]
![[Pasted image 20260426001734.png]]




