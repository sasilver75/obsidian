Microsoft's tile addressing system 

[[Bing QuadKey]], by Microsoft: Microsoft's *tile addressing system* for Bing Maps, based on a [[QuadTree]]
- Just the string of digits tracing the path down the tree
	- The parent of tile "213" is "21", and its children are "2130", "2131", "2132", "2133", with:
		- 0: Top Left
		- 1: Top right
		- 2: Bottom left
		- 3: Bottom right
	- Length: The number of digits equals the zoom level

It's NOT really a [[Discrete Global Grid System|DGGS]], it's just a tile *addressing scheme* on a flat mercator projection, not a spherical partition. Inherits [[Mercator]] distortions; instead, it's more comparable to XYZ [[Slippy Map]] coordinates than to somethingl ike [[H3]] or [[S2 Geometry|S2]]; it's just for *displaying maps, not analyzing spatial data*