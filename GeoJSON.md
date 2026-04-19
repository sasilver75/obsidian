[[JSON]] with a geographic extension.
This is the lingua franca of [[Geographic Information Systems|GIS]]!  It

- ==Use when==: ***Serving data to frontend map***, for small-to-medium datasets (e.g. <~10MB), since most programming languages can easily write it.
- ==Don't use when==: You have millions of features - raw GeoJSON at that scale will crash browsers and waste bandwidth.
	- GeoJSON is a complete dump of all features in one response: The browser has to parse every feature, hold them all in memory, and draw them. AT 1M features, that's hundreds of MB -- the browser tab dies.
	- Instead, you should use [[Tile|Vector Tile]]s! Instead of "give me all the features," the map requests only the tiles that cover the current viewport at the current zoom level! If you're looking at Silver Lake (LA neighborhood), you might get 4-9 tiles worth of data, not the entire city. As you pan and zoom, new tires are loaded and older ones get evicted, and the browser never holds more than what's visible.
	- A tile request might look like `GET /tiles/service_requests/12/703/1651.pbf`, where `12/703/1651` are `zoom/x/y` -- A specific geographic tile at zoom 12.
		- The server would take that, compute the bounding box for that tile, and then run a spatial query against PostGIS to get only the features that intersect it, encoding them as a compact binary [[Protobuf]], and returning 50-200KB instead of 500MB.
			-   (There's also a zoom-level filtering trick: at zoom 10 (city overview), pg_tileserv can return aggregated clusters or simplified geometries. At zoom 15 (street level), it returns full detail. You configure this with SQL views that change behavior based on zoom — so the map is always showing an appropriate level of detail without overwhelming the client. )                                                                        

==TLDR==: Use GeoJSON for small, bounded queries (e.g. "Give me the 50 nearest incidents to this click"), and vector tiles for "show me the whole dataset on a map."


```json
{
	"type": "FeatureCollection",
	"features": [
		{
			"type": "Feature",
			"geometry": {
				"type": "Point",
				"coordinates": [-118.2437, 34.0522]
			},
			"properties": {
				"name": "City Hall",
				"address": "200 N Spring St"
			}
		}
	]
}
```
- Above: See that this GeoJSON object has a type and features, where each feature here has a type, geometry, and properties.

