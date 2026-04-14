Founded in 2010
Mapbox essentially created the modern web mpaping stack
- Built the original [[Mapbox GL JS]] library
- Built the [[Mapbox Vector Tile]] vector tile specification (XYZ tiles)
- Built the style specification that [[MapLibre]] still uses today.
- Form most of the 2010s, they were the go-to choice for anyone building serious map applications.

In December 2020, they changed the license for [[Mapbox GL JS]] v2 from [[BSD License]] to a proprietary license, making it illegal to use without a Mapbox account.
- This was seen as a betrayal of the OS community that had contributed to the project, and it triggered the [[MapLibre]] fork, [[MapLibre GL JS]].


Tiles vs Library:
- Even if we aren't using the MapLibre GL JS library, we can still use the Mapbox tiles and styles.
	- ==The library and the tile service are separate products==
	- ==Mapbox's tiles are still arguably the highest quality available; their catography team is world-class.==

Styles:
- Mapbox has several dark styles:
	- `mapbox://styles/mapbox/dark-v11` is the ==most polished dark map available anywhere!==
	- Streets are visible but muted, labels are clean, and the overall palette is designed to let data overlays pop.

Tradeoffs:
- +: Best cartography, most polished styles, largest ecosystem of tools and documentation.
- +: The standard most developers know - easy to find help.
- -: Requires credit card
- -: License controversy
- -: Vendor lock-in risk (will terms change again?)

