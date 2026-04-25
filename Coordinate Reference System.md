---
aliases:
  - CRS
---

- Longitude: East/West position, range -180 to 180 (negative = west of Greenwich)
- Latitude: North/South position, range -90 to 90 (negative = south of equator)

Los Angeles is roughly (-118.25, 34.05) in (lon, lat) order!
The most common coordinate system shown above is [[WGS84]].
#### Gotcha!
- While [[GeoJSON]], [[Mapbox]], and ==most web APIs use `(longitude, latitude)`...==
- Many *older* tools, [[Shapefile]] conventions, and humans that say "lat/lon" will instead use `(latitude, longitude)`. This is a frequents source of coordinates ending up in the ocean 😄.


### [[Projection]]s
- The Earth is spheroid, and a map is flat.
- A Projection is the mathematical transformation between the two, and **all projections distort something!** (area, shape, distance, or direction!)
- [[EPSG Code]]s are numeric identifiers for coordinate reference systems (CRS). The two that we'll commonly use are:

| EPSG | Name                          | Used for                                                                                   |
| ---- | ----------------------------- | ------------------------------------------------------------------------------------------ |
| 4326 | WGS84 Geographic              | Storing coordinates, GeoJSON, GPS output                                                   |
| 3857 | Web Mercator                  | Displaying [[Tile]]s in web maps (Google Maps, [[Mapbox]], [[OpenStreetMap]] all use this) |
| 2229 | California State Plate Zone 5 | Accurate distance/area calculations in LA, specifically                                    |
- When we store data (using EPSG:4236), it's universal and what every API gives you!
- When you display on a web map, the *tile renderer* handles the 4326 -> 3857 conversion for you!
- When you calculate distances or areas, we project to a local CRS first, or use [[PostGIS]]'s `geography` type (which handles spherical calculations automatically); calculating distance in EPSG:4326 using Euclidean math instead would give you meaningless degree-units.

![[Pasted image 20260411132018.png]]




GPT Image 2
![[Pasted image 20260424234756.png]]
Above:
- Geodetic vs Geocentric:
	- ==Geocentric coordinates==: Origin is the center of the earth, and position expressed as ==(X,Y,Z)== in a cartesian system. Useful for satellite math and physics, the Earth is just a mass at the origin.
	- ==Geodetic coordinates==: The origin is the surface of the ellipsoid, effectively; the position is expressed as ==(latitude, longitude, height)==. This is what GPS gives you, and what humans use for "where on Earth." Note that geodetic latitude IS NOT geocentric latitude; because the Earth is an oblate spheroid (flattened at hte poles), and the surface normal at a point doesn't always point towards the Earth's center, it tilts slight outward to the equator. This matters for precision work!
		- [[WGS84]] (what GPS uses) uses geodetic coordinates: [[EPSG Code|EPSG]]:4236.
