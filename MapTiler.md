A Swiss company, but one of the oldest players in the geospatial software space.
- Originally built tools for creating and hosting custom map tiles, but evolved into a full ==tile hosting platform==. 
- They're one of the primary corporate ==sponsors== of the [[MapLibre GL JS]] project, employing core maintainers and actively contributing to the OS library.
- They also ==create and maintain [[OpenMapTiles]], the open schema that most non-[[Mapbox]] tile providers use.==
	- When you use [[Stadia Maps|Stadia]], [[Protomaps]], or self-hosted tiles, they're almost certainly using the [[OpenMapTiles]] schema.

Styles:
- The "Dataviz" style is explicitly designed for visualization use cases (more neutral cartography that doesn't compete with your data layer)
- Their dark styles include "Streets Dark" and the Dataviz variant.
	- Not quite as nice as [[Stadia Maps|Stadia]]'s Alidade Smooth Dark.

Pricing:
- Free tier for 100,000 requests a month, no CC.
- Paid starts @ $25/month

Tradeoffs:
- +: Deep [[MapLibre]] alignment
- +: Good documentation
- + OpenMapTiles schema means it's easy to switch to self-hosted later.
- -: Smaller free tier than [[Stadia Maps|Stadia]]