---
aliases:
  - OGC Simple Features
  - OGC SFA
---
A widely adopted [[Open Geospatial Consortium]] (OGC) standard for storing and accessing 2D [[Vector]] geospatial data (points, lines, polygons) in databases and GIS software.

Key aspects:
- Geometry Types (Point, LineString, Polygon, Multipoint, MultiCurve, MultiSurface)
- Defines how geometries are represented in text ([[Well-Known Text|WKT]]) and binary ([[Well-Known Binary|WKB]]) formats for data and exchange.
- Defines spatial functions for databases, prefixed with `ST_` (e.g. `ST_Intersects`)

Used extensively by databases like [[PostGIS]]


![[Pasted image 20260425224942.png]]

