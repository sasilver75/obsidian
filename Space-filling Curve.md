A path through space that visits every point in a 2D (or higher) area, while remaining a continuous 1D curve.
- An algorithm that translates two- or n-dimensional data into one-dimensional data. 
- Useful as part of [[Geospatial Index]]es to group vector geometries nearby within a file according to their two-dimensional location, or as part of [[Discrete Global Grid System|DGGS]]s like [[S2 Geometry|S2]] to make nearby cells be nearby in ID.

The key idea: You're creating a mapping from a 1D line to a 2D space that preserves notion of locality. 

Classic examples:
- [[Hilbert Curve]]
- Z-Order (Morton) Curve
- Peano Curve

![[Pasted image 20260416182630.png|500]]
Above: An illustration of the S2 curve after 5 levels of subdivision; In [[S2 Geometry|S2]], six Hilbert curves are linked together to form a single continuous loop over the entire sphere.



Generally, the whole reason that spatial indexing in regular databases is feasible at scale is largely thanks to the space-filling curves that map 2D problems onto 1D indexes, ideally with nice properties like locality preservation (where things nearby in real space are also nearby on the index).