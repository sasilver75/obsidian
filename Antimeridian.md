The antimeridian is the 180 degree line of longitude, the line on the opposite side of the Earth from the [[Prime Meridian]] (0 degree). 
- It runs through the Pacific Ocean and is roughly where the [[International Date Line]] sits.

## Why it causes problems:
- Geographic coordinate systems often represent latitude as a number from -180 to +180; the antimeridian is the *seam* where -180 and 180 meet; they're the same physical line but opposite ends of the numeric range.
	- This causes a specific failure mode: a geometry that crosses the antimeridian gets nonsensical coordinates if you're not careful.
		- Common for something like a bounding box around Fiji, which straddles the antimeridian.
	- Can cause problems for:
		- [[GeoJSON]]: Spec says coordinates are in [[WGS84]] and doesn't say how to handle antimeridian crossing.
		- Bounding boxes
		- Spatial indexes: [[R-Tree]]s and similar structures built on Cartesian assumptions fail to correctly index geometries that wrap around.


Standard solutions:
- Split a geometry at the antimeridian, recommended by RFC 7946 for GeoJSON.











