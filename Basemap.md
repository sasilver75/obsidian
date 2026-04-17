A basemap is a *pre-rendered* or *pre-styled* background map that provides geographic context:
- Roads
- Place names
- Terrain
- Water bodies
- etc.

Your data (hexagons, ship detections, flood polygons, etc.) sits on top of it as layers.

Types of basemaps:
- [[Raster Tile]] basemaps
	- The ==traditional approach==; pre-rendered [[Portable Network Graphics|PNG]] or [[Joint Photographic Experts Group|JPEG]] images served in a grid of 256x256 or 512x512px tiles at each zoom level (XYZ tile scheme).
	- Simple to implement, universally supported, but fixed styling -- you can't change the colors or hide roads without re-rendering everything.
- [[Vector Tile]] basemaps
	- The ==modern approach==; Tiles contain raw vector geometries and attributes (in [[Mapbox Vector Tile]] formats), and styling happens in the client.
	- Tools like [[MapLibre GL JS]] download the vectors and render them in [[WebGL]] in the fly; this means you can restyle the map, hide layers, adjust label language, or even extrude buildings in 3D, all without touching the server.

Common providers:
- Free
	- [[OpenStreetMap]]
	- [[Stadia Maps]]
	- [[Protomaps]] ([[PMTiles]])
	- [[MapTiler]]
- Commercial
	- [[Mapbox]]
	- [[Google Maps]], [[Apple Maps]]
	- [[ESRI]]/[[ArcGIS]] basemaps
	- [[HERE Technologies|HERE]]

Imagery basemaps (satellite/aerial)
- ESRI World Imagery (widely used free satellite basemap in many GIS tools)
- Mapbox Satellite ([[Maxar]] imagery stitched into a global mosaic)
- Google Satellite
- Bing Aerial
