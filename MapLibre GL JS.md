An open-source base-map renderer for rendering interactive maps in the browser with [[WebGL]], handling:
- Fetching [[Vector Tile]] or [[Raster Tile]] data from a [[Tile Server]]
	- Typically using Raster data for satellite imagery (eg.g. a base) and Vector layers on top for roads, labels, boundaries. WebGL renderer handles both types in the same pipeline.
		- If anything, MapLibre GL JS's differentiating strengths are on the vector side (dynamic styling, 3D, data-driven rendering), but raster support is first-class, not an afterthought.
- Parsing and rendering the ==base map== (roads, buildings, water, labels)
	- Note that base maps can be either [[Vector]] or [[Raster]]. Raster is tpically used for Satellite imagery basemaps, but Vector basemaps are usually the modern default because of their styling flexibility and performance benefits.
- Camera: pan, zoom, rotate, tilt
- Coordinate system: Converting long/lat to screen pixels
- A style system for controlling how map features work (defined in a JSON file that describes each layer of the map)

When you pan or zoom, ==MapLibre GL JS fetches the appropriate vector tiles (often in [[Mapbox Vector Tile|MVT]] format (in [[Protobuf]])), decodes them, and renders each layer via WebGL shaders.==
- Maintains a viewport state in `mapStore.ts`; when the user pans, the viewport updates and MapLibre rerenders:
	- center (lng, lat)
	- zoom level
	- bearing (rotation)
	- pitch (tilt)
- Uses [[Mercator|Web Mercator]], ([[EPSG Code|EPSG]]:3857) the standard for web maps.
	- It projects the spherical earth onto a flat square, distorting areas at high latitudes, but preserving shape locality and making math simple.
- 

[[Mapbox]] changed their license in 2020 and started requiring paid tokens for usage, so MapLibre is the community open-source fork that kept the old license. 


MapLibre JS handles:
- Fetching map tiles from a [[Tile Server]] (CARTO, [[Mapbox]], [[OpenStreetMap]], etc.)
- Rendering them as we pan and zoom
- Allowing interactions like pan, scroll to zoom, click events.

Using it together with [[deck.gl]], which will render the data layers on top of the map (pins, POIs, etc.)


MapLibre transparently manages the display of both basema and data layers, regardless of if you're using vector or raster basemap tiles:
- ==Raster Basemap + Vector Data Overlay==
	- When you pan to a new viewport, Maplibre fires off parallel requests for every ==source== in your style:
		- ```
		   GET /satellite/{z}/{x}/{y}.png     ← raster basemap tile
		  GET /data/{z}/{x}/{y}.mvt          ← your vector data tile
		  ```
	- Once both arrive, the WebGL render composites them in layer order from the style spec, raster layer first, then vector layers on top.
	- If one arrives before the other, MapLibre renders what it has and fills in the rest when it arrives. The raster and vector tiles don'to even need to know about eachother at all, MapLibre composes them.
- ==Vector Basemap + Vector Data Overlay==
	- Same thing,but the baemap is also an MVT source instead of Raster.
	```
	GET /basemap/{z}/{x}/{y}.mvt ← vector basemap tile (roads, buildings, etc.)
	 GET /data/{z}/{x}/{y}.mvt ← your vector data tile
	```
	- Both are decoded on the GPU and rendered in layer order. You can interleave layers from both sources. You might choose to render your data layer above water but below road labels, which is something you can't do with raster basemaps, because everything on the raster is already composited together.

Aside:
- Why use MapLibre over Google Maps?
	- Cost. Google Maps charges per map load after a free tier; a public-facing site with real traffic will hit billing quickly, while MapLibre is open source and free.
		- You pay for the tile provider (the actual map imagery), but there several good free options (CARTO, Stadia Maps) that work fine.
	- Control. With Google Maps, you can't change how streets are styled, can't run it offline, and Google can change API pricing at any time. MapLibre has fully customizable styles and runs entirely in your browser.


## Interactivity features
- MapLibre has some built-in interactivity features, like clicking a feature, hovering, etc. but for drawing, it's generally limited and tools like `MapLibre-GL Draw` and `Turf.js` are where you'd want to look.


![[Pasted image 20260425021050.png|1955]]