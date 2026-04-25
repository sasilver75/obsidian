A Python library for manipulating and analyzing geometries.
A foundational tool of the Python geospatial stack!
- Note: Shapely 2.0 (released 2023) was a major rewrite; most tutorials you find are for 1.x; be aware of the API differences.

Wraps the [[GEOS]] C library and ==lets you operate on geometric shapes in 2D space.==

It's important to know:
- It supports ==planar== operations, ==not geographic==!
- It works in raw coordinate space, and doesn't know about [[Coordinate Reference System|Coordinate Reference System]]s or the curvature of the earth!
	- If you can `.distance()` on lat/lon coordinates, you get *degrees*, not *meters*
	- To get real-world measurements, you'd need to either project first (e.g. using [[PyProj]]) or use [[PostGIS]]'s `::geography` cast.

```python
from shapely.geometry import Point, Polygon, LineString

point = Point(−118.25, 34.05)          # a single coordinate
poly  = Polygon([(0,0), (1,0), (1,1), (0,1)])  # a closed ring
line  = LineString([(0,0), (1,1), (2,0)])
```

Common operations:
```
	┌───────────────────┬───────────────────────────────────────┐
    │     Operation     │             What it does              │
    ├───────────────────┼───────────────────────────────────────┤
    │ a.contains(b)     │ Does geometry a fully contain b?      │
    ├───────────────────┼───────────────────────────────────────┤
    │ a.intersects(b)   │ Do they overlap at all?               │
    ├───────────────────┼───────────────────────────────────────┤
    │ a.intersection(b) │ Returns the overlapping region        │
    ├───────────────────┼───────────────────────────────────────┤
    │ a.union(b)        │ Merges two geometries                 │
    ├───────────────────┼───────────────────────────────────────┤
    │ a.distance(b)     │ Distance between nearest points       │
    ├───────────────────┼───────────────────────────────────────┤
    │ a.buffer(100)     │ Expand by 100 units in all directions │
    ├───────────────────┼───────────────────────────────────────┤
    │ a.centroid        │ Returns the center point              │
    ├───────────────────┼───────────────────────────────────────┤
    │ a.area, a.length  │ Geometric measurements                │
    └───────────────────┴───────────────────────────────────────┘
```
