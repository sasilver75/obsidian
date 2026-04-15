An open-source base-map renderer, handling:
- Fetching [[Vector Tile]] data from a [[Tile Server]]
- Parsing and rendering the base map (roads, buildings, water, labels)
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

[[Mapbox]] changed their license in 2020 and started requiring paid tokens for usage, so MapLibre is the community open-source fork that kept the old license. 

It's a TypeScript library used for interactive [[Vector Tile]] maps in the browser!
- Uses GPU-accelerated vector tile rendering.

I'm using it to render the "base map" for my LA information project -- the thing that looks like Google Maps (streets, buildings, outlines, neighborhood labels, satellite imagery).

MapLibre JS handles:
- Fetching map tiles from a [[Tile Server]] (CARTO, [[Mapbox]], [[OpenStreetMap]], etc.)
- Rendering them as we pan and zoom
- Allowing interactions like pan, scroll to zoom, click events.


Using it together with [[deck.gl]], which will render the data layers on top of the map (pins, POIs, etc.)


Aside:
- Why use MapLibre over Google Maps?
	- Cost. Google Maps charges per map load after a free tier; a public-facing site with real traffic will hit billing quickly, while MapLibre is open source and free.
		- You pay for the tile provider (the actual map imagery), but there several good free options (CARTO, Stadia Maps) that work fine.
	- Control. With Google Maps, you can't change how streets are styled, can't run it offline, and Google can change API pricing at any time. MapLibre has fully customizable styles and runs entirely in your browser.