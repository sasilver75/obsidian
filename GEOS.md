[[GDAL]], [[GEOS]], and [[PROJ]]  are three C libraries that underpin almost all geospatial software. You rarely call them directly; they're the engine under tools like [[PostGIS]], [[Shapely]], [[QGIS]], and more. GEOS implements the JTS (Java Topology Suite) specification in C.

==GEOS handles computational geometry operations on 2D vector features.==

Given two polygons, GEOS answers questions like:
- Do they intersect? (`ST_INTERSECTS()`)
- What is their union? (`ST_UNION()`)
- What is their intersection? (`ST_INTERSECTION()`)
- Is point A inside polygon B? (`ST_CONTAINS()`)
- What is the distance between these two geometries? (`ST_DISTANCE()`)

==All of [[PostGIS]]'s spatial relationship function are GEOS==.

==[[Shapely]] is essentially a Python wrapper around GEOS==:
When you call `shapely.geometry.Point(lon,lat).within(polygon)`, it's GEOS that's actually doing the math.


