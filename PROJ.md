[[GDAL]], [[GEOS]], and [[PROJ]]  are three C libraries that underpin almost all geospatial software. You rarely call them directly; they're the engine under tools like [[PostGIS]], [[Shapely]], [[QGIS]], and more.

PROJ ==converts coordinates between [[Coordinate Reference System|Coordinate Reference System]]s==.
- The earth is a weird lumpy ellipsoid; every map projection is a mathematical function that flattens some portion of it onto a 2D plane.
- PROJ knows hundreds of these functions and can transform a point from one to another.
- GPS might give you WGS84, but City survey data might be in a local projection, while satellite imagery might be in UTM. You need PROJ to put them all in the same system!

    WGS84 (latitude/longitude) → California State Plane (meters, local accuracy)
    EPSG:4326 → EPSG:2229

[[PostGIS]]'s `ST_TRANSFORM()` function called PROJ under the hood!