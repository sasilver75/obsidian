---
aliases:
  - OSM
---
A free, editable, openly-licensed map of the entire world built by volunteers -- essentially ==the Wikipedia of maps== with a highly permissive license.
- Mostly, it's purely [[Vector]] data (nodes, ways, and relations with tags), though the OSM ecosystem has grown to include some raster-adjacent things.

==The central tension of OSM is that data quality is highly variable==. 
- Dense urban areas in wealthy countries often have better coverage than even commercial alternatives (e.g. London, Berlin, New York).
- Rural Areas in developing countries can be sparse or outdated, though.
- Schema inconsistency; Because anyone can add a tag, the same feature might be tagged 5 different ways by different contributors ("building=yes", "building=house", "building=residential").
	- This is one of [[Overture Maps Foundation|Overture Maps]]'s main criticisms of OSM.

The OSM project provides free [[Raster Tile]]s at `tile.openstreetmap.org`
- These are pre-rendered PNG images at each zoom level; the classic [[Slippy Map]] tiles that powered Google Maps-style web maps before [[Vector Tile]]s existed.

Fundamental limitation
- Raster tiles are images, not data. 
	- [[MapLibre GL JS]] can render them, but you lose everything that makes vector tiles powerful:
		- You can't change the style
		- Can't change the label language
		- Can't animate
		- Can't control layer ordering
	- The hex layer would sit on top of a fixed image, rather than integrating into the map.
- Also, OSM's usage policy proohibits heavy use in applications; they're met for low-traffic OSM ecosystem projects, not production applications.

Tradeoffs:
- +: Free, no key, no signup
- -: Raster, can't customize style to complement hex layer
- -: Usage polixy prohibits production use
- -: Lower resolution appearance vs vector tiles


  Data access:
  - Planet.osm — full planet dump, updated weekly (~80GB compressed). Everything.
  - Geofabrik — pre-cut regional extracts (country, state level) in PBF or shapefile
  format. The standard way to get OSM data for a specific region.
  - Overpass API — query OSM data by tag/location without downloading a full extract.
  osmnx wraps this for Python.
  - osmnx — Python library that fetches OSM road networks and building footprints via
  Overpass, directly into NetworkX graphs or GeoDataFrames. Extremely useful.



  ***Humanitarian OSM Team (HOT)***
  - A standout sub-organization. After the 2010 Haiti earthquake, OSM volunteers remotely mapped Port-au-Prince from satellite imagery in 48 hours — producing the most detailed map of the city that had ever existed, used by relief organizations on the ground.
  - HOT now runs organized mapping campaigns for disaster response and
  development work globally. It demonstrated that crowdsourced mapping could outpace any government or commercial effort in a crisis.

> Q: Why not just use like... Google Maps or Apple Maps or something?
> A: Licensing; Google Maps and Apple Maps data is proprietary, and you can't download/redistribute/use as a base doe your own derived dataset. Relief organizations need to be able to share data across agencies, governments, and NGOs without legal encumbrance. 
> A: Offline use; Relief workers can download OSM data nd use it completely offline in the field.
> A: Editability; You can't add a field hospital, refugee camp, newly-cleared road, or a damaged bridge to their database.
> A: Coverage of unmapped places. Before the HAiti earthquake, if you can believe it, Google Maps had almost nothing for Port-au-Price at street level. Commercial map providers focus coverage investment on wealthy markets with high ad revenue potential.
> A: Cost; Google Maps API calls cost money at scale.
>
> TLDR: Google Maps is a consumer product optimized for ad-supported navigation in wealthy markets; OSM is a database optimized for open reuse; they solve different problems. 


