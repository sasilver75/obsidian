---
aliases:
  - S2
---
A library and [[Discrete Global Grid System]] developed by Google (used in [[Google Maps]], [[BigQuery]], others) for representing and indexing geographic regions on a sphere.
Unlike systems that work on flat 2D projections, S2 treats the earth as an actual sphere.

Core idea:
- S2 maps the sphere onto the six faces of a cube, then subdivides each face into a hierarchical grid of cells, with each cell having a ==unique 64-bit integer ID==, called an ==S2 Cell ID==
	- The 64-bit integer ID means that you can index spatial data in a regular spatial index like a [[B-Tree]] without needing a specialized spatial index like [[R-Tree]] or [[GiST]], making it popular for large-scale systems.

Cell hierarchy:
- 30 levels of subdivision; Level 0 = one of 6 cube faces, level 30 = ~1cm^2
- Each cell has exactly 4 children
- Cells at the same level are roughly equal area
- A cell ID encodes both its position and its level

What you can do with it:
- Point indexing: Convert lat/lng to a cell ID for fast lookup
- Region coverage: Approximate any shape (polygon, circle, etc) with a set of cells
- Proximity queries: Find points near a location efficiently
- Range queries: Nearby cells have nearby IDs ([[Space-filling Curve]] property)


![[Pasted image 20260416172435.png]]
Above: Showing two of the six "face cells" in S2, one of which has been subdivided several times.

![[Pasted image 20260416182630.png|500]]
Above: An illustration of the S2 curve after 5 levels of subdivision; In [[S2 Geometry|S2]], six Hilbert curves are linked together to form a single continuous loop over the entire sphere.


![[Pasted image 20260416183440.png]]
Above: Not fully accurate, but conceptually correct illustration of the S2 curve on the unit cube (before projecting it to the sphere). The cube has been unfolded and flattened, and shows the fifrst 4 levels of the S2 curve subdivision.