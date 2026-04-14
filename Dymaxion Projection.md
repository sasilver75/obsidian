---
aliases:
  - Fuller Projection
---
A map projection invented by Buckminster Fuller in 1943, it's the geometric foundation that [[H3]] is built on!

The core idea:
- Unfold the Earth onto an icosahedron (a 20-sided solid made of equilateral triangles)
- Unfold those triangles flat onto a 2D map.

Key property:
- An ==icosahedron is the Platonic solid that most closely approximates a sphere==, so each triangular surface covers a relatively small area and the ==distortion per face is minimized==.

Most map projections have a fundamental problem that flattening a sphere into a 2D surface causes ==distortion!== 
- [[Mercator|Mercator Projection]] for instance massively distorts area (Greenland looks as big as Africa)

This matters for [[H3]] because if your base projection has uneven distortion, your "equal area" hexagons won't actually be equal area!
- ==The Dymaxion projection distributes distortion more evenly across the globe than any single-perspective projection.==
- It has ==no singular poles==, which are a major source of distortion in cylindrical projections.
- Its ==triangular faces map naturally onto a hexagonal or pentagonal tiling==.

![[Pasted image 20260413205642.png]]

All of the points of it are in the ocean, while we usually care about things on land. 
