---
aliases:
  - WKT
---
Commonly written WKT and pronounced as "Wicket", Well-Known Text, along with [[Well-Known Binary]] (WKB) are serialization formats for geospatial geometry objects -- standardized ways to represent a shape as a string (WKT) or binary blob (WKB).

WKTs are human-readable:
>POINT(-118.2437 34.0522)                                                             
>LINESTRING(-118.24 34.05, -118.25 34.06, -118.26 34.07)                             
>POLYGON((-118.24 34.05, -118.25 34.05, -118.25 34.06, -118.24 34.06, -118.24 34.05)) 
>MULTIPOLYGON(...) 

You'll see WKTs when inspecting [[PostGIS]] data, in GIS desktop tools, and sometimes in API responses. 
It's also how you construct geometries in SQL:
```SQL
ST_GeomFromText('POINT(-118.2437, 34.0522', 4326))
```
Above: See that 4326? That's a reference to [[WGS84|EPSG:4326]], which is the [[EPSG Code]] for [[WGS84]]
